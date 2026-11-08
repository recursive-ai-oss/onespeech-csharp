using System.Collections.Immutable;
using OneSpeech.Catalog;
using OneSpeech.Qwen3;

namespace OneSpeech.Pipeline;

// Load only the models actually needed by a chunk, after recognition has selected its language.
internal sealed class OrderedTimingAligner(ModelStore store, bool realign = false) : Aligner, IDisposable
{
    private Qwen3ForcedAligner? qwen;
    private Alignment.OmniAsrAligner? ctc;
    public override ModelInfo Model => qwen?.Model ?? ctc?.Model ?? throw new InvalidOperationException("No timing model has been loaded.");
    public override Capabilities Capabilities { get; } = new() { ForcedAlignment = true,
        Languages = [.. Qwen3ForcedAligner.SupportedLanguages.Concat(store.Catalog.GetDefault("omniasr-ctc").Capabilities.Languages).Distinct(StringComparer.Ordinal)] };

    internal static bool CanUseQwen(AudioBuffer audio, string? language) =>
        language is not null && Qwen3ForcedAligner.SupportedLanguages.Contains(language) && audio.FrameCount is > 200 and <= 4800000;

    private IEnumerable<Aligner> Candidates(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        if (CanUseQwen(audio, language))
        {
            qwen ??= SpeechServices.TryResolveQwenAlignerAsync(store, language!, cancellationToken).AsTask().GetAwaiter().GetResult();
            if (qwen is not null) yield return qwen;
        }
        if (audio.Duration <= TimeSpan.FromSeconds(60) && (language is null || store.Catalog.GetDefault("omniasr-ctc").Capabilities.Languages.Contains(language, StringComparer.Ordinal)))
        {
            if (qwen is not null && CanUseQwen(audio, language))
            {
                // An occasional fallback must not pin a second model's large shape
                // plan for the rest of a long recording. Iterator disposal also
                // releases it if timing succeeds early, fails, or is canceled.
                using var fallback = SpeechServices.ResolveAlignerAsync(store, null, cancellationToken).AsTask().GetAwaiter().GetResult();
                yield return fallback;
            }
            else
            {
                // Where CTC is the primary timer, reuse it across adjacent chunks.
                ctc ??= SpeechServices.ResolveAlignerAsync(store, null, cancellationToken).AsTask().GetAwaiter().GetResult();
                yield return ctc;
            }
        }
    }

    public override Transcript Align(AudioBuffer audio, string text, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, "ordered timing");
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(w => new Word { Text = w, Language = language }).ToImmutableArray();
        var original = new Transcript { Text = text, Language = language, Duration = audio.Duration, Words = words,
            Segments = [new Segment { Text = text, Words = words, Start = TimeSpan.Zero, End = audio.Duration, Language = language }] };
        return Align(audio, original, language, cancellationToken);
    }

    public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language ?? transcript.Language, Capabilities, "ordered timing");
        // Explicit alignment can target different audio; old timings are not evidence for the new input.
        if (realign) transcript = Alignment.AlignedWords.Replace(transcript,
            transcript.Words.Select(w => w with { Start = null, End = null, TimingMethod = TimingMethod.None }).ToImmutableArray());
        return VadTranscriptionPipeline.ApplyTiming(audio, transcript, language, Candidates(audio, language ?? transcript.Language, cancellationToken), cancellationToken);
    }

    protected override void Dispose(bool disposing) { base.Dispose(disposing); if (!disposing) return; qwen?.Dispose(); ctc?.Dispose(); }
}
