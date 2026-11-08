using System.Collections.Immutable;
using System.Text;
using OneSpeech.Pipeline;
using OneSpeech.Vad;

namespace OneSpeech;

/// <summary>A contextual phrase, not a required output or a replacement rule.</summary>
/// <param name="Text">Nonempty ordinary text, at most 1024 UTF-16 units; special tokens are refused by each tokenizer.</param>
/// <param name="Weight">Optional natural-log bonus per token, in (0,20]. Null selects 1 for fusion.
/// Prompt-conditioned families refuse explicit weights because prompts cannot implement numeric score bonuses.</param>
public sealed record Phrase(string Text, double? Weight = null);

internal static class PhraseRequests
{
    internal static ImmutableArray<Phrase> Copy(IEnumerable<Phrase> phrases, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        var result = ImmutableArray.CreateBuilder<Phrase>(); int chars = 0;
        foreach (var phrase in phrases)
        {
            ct.ThrowIfCancellationRequested();
            if (phrase is null || string.IsNullOrWhiteSpace(phrase.Text) || phrase.Text.Length > 1024 ||
                phrase.Weight is double w && (!double.IsFinite(w) || w is <= 0 or > 20) ||
                result.Count == 4096 || (chars += phrase.Text.Length) > 262144)
                throw new ArgumentException("Invalid contextual phrase, weight or list bound.", nameof(phrases));
            try { _ = new UTF8Encoding(false, true).GetByteCount(phrase.Text); }
            catch (EncoderFallbackException ex) { throw new ArgumentException("A phrase has invalid UTF-16.", nameof(phrases), ex); }
            result.Add(phrase with { Text = phrase.Text.Trim() });
        }
        ct.ThrowIfCancellationRequested(); return result.ToImmutable();
    }

    internal static string Prompt(ImmutableArray<Phrase> phrases)
    {
        if (phrases.Any(p => p.Weight.HasValue)) throw new CapabilityNotSupportedException("Prompt conditioning does not support numeric phrase weights; omit Weight.");
        return string.Join(", ", phrases.Select(p => p.Text));
    }
}

// Borrows model ownership; context is immutable per wrapper. The model's concurrency restrictions still apply.
internal sealed class BiasedRecognizer(SpeechRecognizer source, Func<AudioBuffer, string?, CancellationToken, Transcript> transcribe,
    Func<AudioBuffer, string?, CancellationToken, IEnumerable<Segment>>? stream = null) : SpeechRecognizer
{
    // The wrapper borrows the source; only the original recognizer owns its model resources.
    protected override void Dispose(bool disposing) => base.Dispose(disposing);
    public override ModelInfo Model => source.Model;
    public override Capabilities Capabilities => source.Capabilities;
    public override SpeechRecognizer WithPhrases(IEnumerable<Phrase> phrases, CancellationToken cancellationToken = default) => source.WithPhrases(phrases, cancellationToken);
    public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(audio); cancellationToken.ThrowIfCancellationRequested();
        if (stream != null || audio.FrameCount <= 480000) return transcribe(audio, language, cancellationToken);
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        return new VadTranscriptionPipeline(this, vad).Transcribe(audio, language, cancellationToken);
    }
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(audio); cancellationToken.ThrowIfCancellationRequested();
        if (stream != null) { foreach (var segment in stream(audio, language, cancellationToken)) yield return segment; yield break; }
        if (audio.FrameCount <= 480000) { foreach (var segment in transcribe(audio, language, cancellationToken).Segments) yield return segment; yield break; }
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        foreach (var segment in new VadTranscriptionPipeline(this, vad).TranscribeStream(audio, language, cancellationToken)) yield return segment;
    }
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken = default) => source.DetectLanguage(audio, cancellationToken);
}
