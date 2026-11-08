using System.Collections.Immutable;
using OneSpeech.Pipeline;
using OneSpeech.Vad;
using RecursiveAI.OneModel.Graph;

namespace OneSpeech.Qwen3;

/// <summary>Offline Qwen3-ASR with native OneModel audio encoder and cached language decoder. Instances are not concurrent.</summary>
/// <remarks>Input must be 16 kHz mono. Longer input uses VAD-first 30-second chunks. Word timings remain null until an aligner supplies them.</remarks>
public sealed class Qwen3Recognizer : SpeechRecognizer, IDisposable
{
    private readonly QwenModelSession runtime;
    private readonly Qwen3Tokenizer tokenizer;
    private readonly Qwen3Options options;
    private bool disposed;
    internal Aligner? WordAligner { get; set; }
    private Qwen3Recognizer(string directory, Qwen3Options options, CancellationToken cancellationToken)
    {
        options.Validate(); this.options = options;
        tokenizer = new Qwen3Tokenizer(directory);
        if (tokenizer.Encode(options.Context, cancellationToken: cancellationToken).Length > 4096) throw new ArgumentException("Qwen context exceeds 4096 tokens.", nameof(options));
        if (new FileInfo(Path.Combine(directory, "upstream.txt")).Length > 4096) throw new ModelIntegrityException("Qwen provenance exceeds 4 KiB.");
        string[] pin = File.ReadAllLines(Path.Combine(directory, "upstream.txt"));
        if (pin.Length < 2) throw new ModelIntegrityException("Missing Qwen converter provenance.");
        string expected = pin[0] switch
        {
            "Qwen/Qwen3-ASR-0.6B" => "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
            "Qwen/Qwen3-ASR-1.7B" => "7278e1e70fe206f11671096ffdd38061171dd6e5",
            _ => throw new ModelIntegrityException("Unsupported Qwen ASR checkpoint.")
        };
        if (pin[1] != expected) throw new ModelIntegrityException("Qwen checkpoint revision differs from the supported pin.");
        Model = new ModelInfo { Id = pin[0], Source = "https://huggingface.co/" + pin[0], Revision = expected, License = "Apache-2.0", Tier = ModelTier.Tier1, Attribution = "Qwen Team, Qwen3-ASR, Apache License 2.0.",
            Files = pin[0] == "Qwen/Qwen3-ASR-0.6B"
                ? [new ModelFileInfo { Name = "model.safetensors", Sha256 = "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea" }]
                : [new ModelFileInfo { Name = "model-00001-of-00002.safetensors", Sha256 = "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6" }, new ModelFileInfo { Name = "model-00002-of-00002.safetensors", Sha256 = "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc" }] };
        if (options.Precision == Qwen3Precision.BFloat16)
            Catalog.QwenPrecisionAssets.EnsureBFloat16(Catalog.ModelCatalog.Default.Get(pin[0]), directory, cancellationToken);
        runtime = new QwenModelSession(directory, options.Threads, cancellationToken, options.Precision);
    }
    /// <summary>Loads local graphs and tokenizer data. Options default to greedy generation, no context, Binary32 and automatic thread selection. A BFloat16 request verifies or lazily converts its graphs from pinned publisher files in this directory; it never downloads. Concurrent verification or conversion of the same directory throws IOException.</summary>
    /// <exception cref="IOException">A model file is unavailable.</exception>
    /// <exception cref="ModelIntegrityException">The converter provenance is unsupported.</exception>
    /// <exception cref="ArgumentException">Options or tokenizer input exceed supported bounds.</exception>
    /// <exception cref="OperationCanceledException">Loading is canceled.</exception>
    public static Qwen3Recognizer Load(string directory, Qwen3Options? options = null, CancellationToken cancellationToken = default) => new(directory, options ?? new Qwen3Options(), cancellationToken);
    /// <inheritdoc/>
    public override ModelInfo Model { get; }
    /// <inheritdoc/>
    public override Capabilities Capabilities { get; } = new() { Languages = Qwen3Languages.Tags, Biasing = true };

    /// <inheritdoc/>
    public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(audio);
        cancellationToken.ThrowIfCancellationRequested();
        if (language != null) _ = Qwen3Languages.Name(language);
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new UnsupportedAudioException("Qwen3-ASR requires 16000 Hz mono PCM; resample/downmix through OneAudio.");
        if (audio.FrameCount > 480000)
        {
            using var vad = new SileroVad(cancellationToken: cancellationToken);
            return new VadTranscriptionPipeline(this, vad).Transcribe(audio, language, cancellationToken);
        }
        if (audio.FrameCount == 0) return new Transcript { Text = "", Language = language, Models = [Model] };
        if (audio.FrameCount <= 200) throw new UnsupportedAudioException("Qwen3-ASR unpadded input must exceed 200 samples.");
        using var encoded = runtime.Encode(audio, cancellationToken);
        int count = encoded["encoder"].Data<float>().Length / runtime.Configuration.Width;
        string prefix = "<|im_start|>system\n" + options.Context + "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
        var prompt = tokenizer.Encode(prefix, true, cancellationToken).ToList();
        for (int i = 0; i < count; i++) prompt.Add(151676);
        prompt.AddRange(tokenizer.Encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n" + (language == null ? "" : "language " + Qwen3Languages.Name(language) + "<asr_text>"), true, cancellationToken));
        SessionOutputs[]? past = null;
        List<int> generated = [];
        try
        {
            float[] logits = runtime.Decode(prompt.ToArray(), encoded["encoder"], ref past, 0, cancellationToken);
            bool finished = false;
            for (int step = 0; step < options.MaxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int best = 0;
                for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
                if (best == 151645) { finished = true; break; }
                generated.Add(best);
                logits = runtime.Decode([best], null, ref past, prompt.Count + step, cancellationToken);
            }
            if (!finished) throw new OneSpeechException("Qwen3-ASR reached MaxNewTokens; increase the per-chunk generation limit.");
        }
        finally { if (past != null) foreach (var output in past) output.Dispose(); }
        string text = tokenizer.Decode(generated.ToArray(), true);
        if (language == null)
        {
            int separator = text.IndexOf("<asr_text>", StringComparison.Ordinal);
            if (separator < 0 || !text.StartsWith("language ", StringComparison.Ordinal)) throw new ModelIntegrityException("Qwen3-ASR did not emit its declared language/text format.");
            string name = text[9..separator];
            language = Qwen3Languages.Tag(name);
            if (language == null && name != "None") throw new CapabilityNotSupportedException("Qwen3-ASR emitted an undeclared language: " + name);
            text = text[(separator + 10)..];
        }
        if (WordAligner != null && language != null && WordAligner.Capabilities.Languages.Contains(language))
        {
            var aligned = WordAligner.Align(audio, text, language, cancellationToken);
            return aligned with { Models = [Model, .. aligned.Models] };
        }
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(word => new Word { Text = word, Language = language, Speaker = SpeakerLabel.Unassigned }).ToImmutableArray();
        var segment = new Segment { Text = text, Start = TimeSpan.Zero, End = audio.Duration, Language = language, Words = words };
        return new Transcript { Text = text, Language = language, Duration = audio.Duration, Words = words, Segments = [segment], Models = [Model] };
    }
    /// <inheritdoc/>
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this); ArgumentNullException.ThrowIfNull(audio); cancellationToken.ThrowIfCancellationRequested();
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new UnsupportedAudioException("Qwen3-ASR requires 16000 Hz mono PCM.");
        if (audio.FrameCount <= 480000) { foreach (var segment in Transcribe(audio, language, cancellationToken).Segments) yield return segment; yield break; }
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        foreach (var segment in new VadTranscriptionPipeline(this, vad).TranscribeStream(audio, language, cancellationToken)) yield return segment;
    }
    /// <inheritdoc/>
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => Transcribe(audio, null, cancellationToken).Language ?? throw new UnsupportedAudioException("No language was detected in Qwen3-ASR output.");
    /// <summary>Releases all native-managed model buffers. Safe to call repeatedly; subsequent recognition throws ObjectDisposedException.</summary>
    public void Dispose() { if (disposed) return; disposed = true; runtime.Dispose(); }
}
