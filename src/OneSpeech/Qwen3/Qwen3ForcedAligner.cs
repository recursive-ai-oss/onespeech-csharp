using System.Collections.Immutable;
using System.Text;
using RecursiveAI.OneModel.Graph;

namespace OneSpeech.Qwen3;

/// <summary>Qwen3's non-autoregressive timestamp-slot aligner on OneModel. Instances are not concurrent.</summary>
/// <remarks>Accepts more than 200 samples and at most 300 seconds of 16 kHz mono audio. Language is required.
/// Native timestamp resolution is 80 ms. Invalid predictions retain null timing rather than fabricated boundaries.</remarks>
public sealed partial class Qwen3ForcedAligner : Aligner, IDisposable
{
    private readonly QwenModelSession runtime;
    private readonly Qwen3Tokenizer tokenizer;
    private bool disposed;
    internal static readonly ImmutableArray<string> SupportedLanguages = ["zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"];
    private Qwen3ForcedAligner(string directory, int threads, CancellationToken cancellationToken)
    {
        if (threads is < 0 or > 256) throw new ArgumentOutOfRangeException(nameof(threads));
        string path = Path.Combine(directory, "upstream.txt");
        if (new FileInfo(path).Length > 4096) throw new ModelIntegrityException("Qwen provenance exceeds 4 KiB.");
        string[] pin = File.ReadAllLines(path);
        if (pin.Length < 2 || pin[0] != "Qwen/Qwen3-ForcedAligner-0.6B" || pin[1] != "c7cbfc2048c462b0d63a45797104fc9db3ad62b7")
            throw new ModelIntegrityException("Unsupported Qwen forced-aligner provenance.");
        tokenizer = new Qwen3Tokenizer(directory);
        runtime = new QwenModelSession(directory, threads, cancellationToken);
        if (runtime.Configuration.TimestampClasses != 5000) { runtime.Dispose(); throw new ModelIntegrityException("Expected the released 5000-class Qwen timestamp head."); }
    }
    /// <summary>Loads locally converted graphs and model data. Threads default to 1; zero selects runtime automatic; 0–256 are accepted.</summary>
    /// <exception cref="IOException">A local model file is missing or unreadable.</exception>
    /// <exception cref="ModelIntegrityException">Checkpoint provenance or classifier dimensions are unsupported.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thread count is outside its bounds.</exception>
    /// <exception cref="OperationCanceledException">Loading is canceled.</exception>
    public static Qwen3ForcedAligner Load(string directory, int threads = 1, CancellationToken cancellationToken = default) => new(directory, threads, cancellationToken);
    /// <inheritdoc/>
    public override ModelInfo Model { get; } = new() { Id = "Qwen/Qwen3-ForcedAligner-0.6B", Source = "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B", Revision = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7", License = "Apache-2.0", Tier = ModelTier.Tier1, Attribution = "Qwen Team, Qwen3-ForcedAligner, Apache License 2.0.", Files = [new ModelFileInfo { Name = "model.safetensors", Sha256 = "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7" }] };
    /// <inheritdoc/>
    public override Capabilities Capabilities { get; } = new() { Languages = SupportedLanguages, NativeTimestamps = true, ForcedAlignment = true };

    /// <summary>Aligns whitespace-delimited written units, or individual Han/Japanese characters, to native 80-ms slots. Text is preserved on the document.
    /// Null or unsupported language throws CapabilityNotSupportedException; invalid audio/oversized input throws ArgumentException.
    /// Cancellation is checked between model graph calls. No embeddings are persisted.</summary>
    public override Transcript Align(AudioBuffer audio, string text, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(audio); ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        if (language == null || !Capabilities.Languages.Contains(language)) throw new CapabilityNotSupportedException("Qwen3-ForcedAligner requires one of its eleven declared language tags.");
        if (audio.SampleRate != 16000 || audio.Channels != 1 || audio.FrameCount is <= 200 or > 4800000) throw new ArgumentException("Qwen3-ForcedAligner accepts 200–4800000 samples (exclusive lower bound) of 16 kHz mono audio.");
        if (new UTF8Encoding(false, true).GetByteCount(text) > 1048576) throw new ArgumentException("Alignment text exceeds 1 MiB UTF-8.", nameof(text));
        var units = Units(text, language).ToArray();
        if (units.Length > 4096) throw new ArgumentException("Qwen alignment exceeds 4096 written units.", nameof(text));
        if (units.Length == 0) return new Transcript { Text = text, Language = language, Duration = audio.Duration, Models = [Model] };
        using var encoded = runtime.Encode(audio, cancellationToken);
        int audioTokens = encoded["encoder"].Data<float>().Length / runtime.Configuration.Width;
        List<int> prompt = [151669];
        for (int i = 0; i < audioTokens; i++) prompt.Add(151676);
        prompt.Add(151670);
        List<int> slots = [];
        foreach (var unit in units)
        {
            prompt.AddRange(tokenizer.Encode(unit.Spoken, cancellationToken: cancellationToken));
            slots.Add(prompt.Count); prompt.Add(151705); prompt.Add(151705);
        }
        SessionOutputs[]? state = null;
        float[] logits;
        try { logits = runtime.Decode(prompt.ToArray(), encoded["encoder"], ref state, 0, cancellationToken); }
        finally { if (state != null) foreach (var item in state) item.Dispose(); }
        var words = ImmutableArray.CreateBuilder<Word>(units.Length);
        (int Index, double Probability) Predict(int position)
        {
            int classes = runtime.Configuration.TimestampClasses, begin = position * classes, best = 0;
            for (int i = 1; i < classes; i++) if (logits[begin + i] > logits[begin + best]) best = i;
            double sum = 0;
            for (int i = 0; i < classes; i++) sum += Math.Exp(logits[begin + i] - logits[begin + best]);
            return (best, 1 / sum);
        }
        for (int i = 0; i < units.Length; i++)
        {
            var start = Predict(slots[i]); var end = Predict(slots[i] + 1);
            TimeSpan begin = TimeSpan.FromMilliseconds(start.Index * 80), finish = TimeSpan.FromMilliseconds(end.Index * 80);
            bool valid = finish >= begin && finish <= audio.Duration;
            words.Add(new Word { Text = units[i].Written, Start = valid ? begin : null, End = valid ? finish : null, Confidence = (start.Probability + end.Probability) / 2,
                TimingMethod = valid ? TimingMethod.ForcedAligner : TimingMethod.None, Language = language, Speaker = SpeakerLabel.Unassigned });
        }
        var result = words.ToImmutable();
        return new Transcript { Text = text, Language = language, Duration = audio.Duration, Words = result, Models = [Model], Segments = [new Segment { Text = text, Start = TimeSpan.Zero, End = audio.Duration, Words = result, Language = language }] };
    }
    /// <summary>Aligns existing written words, preserving their boundaries and non-timing fields. Character slots for Chinese and Japanese are projected back to each written word only when every slot is valid; punctuation-only words remain untimed. Limits and exceptions follow text alignment.</summary>
    public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language ?? transcript?.Language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(transcript);
        language ??= transcript.Language;
        var timed = Align(audio, string.Join(' ', transcript.Words.Select(w => w.Text)), language, cancellationToken);
        int cursor = 0;
        var words = transcript.Words.Select(word =>
        {
            int count = Units(word.Text, language!).Count();
            var slots = timed.Words.AsSpan(cursor, count);
            cursor += count;
            bool valid = count != 0;
            foreach (var slot in slots) valid &= slot.Start.HasValue && slot.End.HasValue;
            return word with { Start = valid ? slots[0].Start : null, End = valid ? slots[^1].End : null,
                TimingMethod = valid ? TimingMethod.ForcedAligner : TimingMethod.None };
        }).ToImmutableArray();
        return Alignment.AlignedWords.Replace(transcript, words) with { Duration = timed.Duration,
            Models = [.. transcript.Models.Concat(timed.Models).DistinctBy(m => (m.Id, m.Revision))] };
    }

    internal static string SpokenForm(string written) => string.Concat(written.EnumerateRunes().Where(rune => Rune.IsLetterOrDigit(rune) || rune.Value == '\'').Select(rune => rune.ToString()));
    private static IEnumerable<(string Written, string Spoken)> Units(string text, string language)
    {
        foreach (string written in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            string spoken = SpokenForm(written);
            if (spoken.Length == 0) continue;
            if (language is "zh" or "yue" or "ja")
            {
                foreach (Rune rune in spoken.EnumerateRunes()) yield return (rune.ToString(), rune.ToString());
            }
            else yield return (written, spoken);
        }
    }
    /// <summary>Releases model buffers. Repeated calls are safe; later alignment throws ObjectDisposedException.</summary>
    protected override void Dispose(bool disposing) { base.Dispose(disposing); if (!disposing) return; if (disposed) return; disposed = true; runtime.Dispose(); }
}
