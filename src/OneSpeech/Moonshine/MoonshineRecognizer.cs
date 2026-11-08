using System.Collections.Immutable;

namespace OneSpeech.Moonshine;

/// <summary>English Moonshine fp32 greedy recognition from independently converted OneModel graphs. Instances are not concurrent; dispose after use. Accepts mono 16 kHz utterances of at most 60 seconds; use the file/live VAD pipeline for longer recordings.</summary>
public sealed partial class MoonshineRecognizer : SpeechRecognizer, IDisposable
{
    private readonly MoonshineModelSession runtime;
    private readonly MoonshineTokenizer tokenizer;
    private readonly MoonshineOptions options;
    private readonly Capabilities capabilities;
    private bool disposed;
    internal Action<int[]>? TokenTrace { get; set; }
    private MoonshineRecognizer(string directory, MoonshineOptions options, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        options.Validate(); this.options = options;
        var entry = MoonshineProvenance.Verify(directory, converted: true, cancellationToken);
        capabilities = entry.Capabilities;
        Model = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, License = entry.License, Attribution = entry.Attribution, Tier = entry.Tier, Files = [.. entry.Files.Concat(entry.ConvertedFiles).Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha256 })] };
        tokenizer = new MoonshineTokenizer(directory);
        runtime = new MoonshineModelSession(directory, options.Threads, cancellationToken);
    }
    /// <summary>Loads pinned offline or streaming-family graphs, tokenizer data and upstream.txt from a local directory. Defaults to one thread, 6.5 new tokens/audio second. Throws typed model/capability errors for unsupported provenance, file/JSON errors for corrupt data, OperationCanceledException when canceled. Does not download models.</summary>
    public static MoonshineRecognizer Load(string directory, MoonshineOptions? options = null, CancellationToken cancellationToken = default) => new(directory, options ?? new(), cancellationToken);
    /// <inheritdoc />
    public override ModelInfo Model { get; }
    /// <inheritdoc />
    public override Capabilities Capabilities => capabilities;
    /// <summary>Creates a VAD-endpointed session. Streaming checkpoints emit replacing partials with cached encoder prefixes and per-hypothesis decoder KV/cross-attention projections; offline checkpoints emit finals only. Streaming session creation preallocates and warms bounded working buffers. Borrows this recognizer and detector; dispose the session before either model.</summary>
    public override LiveSession CreateLiveSession(VoiceActivityDetector detector, string? language = null)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(detector);
        if (language is not (null or "en")) throw new CapabilityNotSupportedException("Moonshine checkpoints support English only.");
        return runtime.Configuration.Streaming ? new Live.UtteranceLiveSession(this, detector, new MoonshineStreamingState(this), language) : base.CreateLiveSession(detector, language);
    }
    /// <inheritdoc />
    public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(disposed, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (language is not (null or "en")) throw new CapabilityNotSupportedException("These Moonshine checkpoints recognize English only; select language 'en' or omit the hint.");
        if (audio.SampleRate != 16000 || audio.Channels != 1 || audio.FrameCount > 960000) throw new ArgumentException("Moonshine requires mono 16000 Hz utterances of at most 60 seconds. Use a VAD pipeline for longer input.");
        foreach (float sample in audio.Samples.Span) if (!float.IsFinite(sample)) throw new ArgumentException("Moonshine requires finite PCM samples.");
        if (audio.FrameCount == 0) return new Transcript { Text = "", Duration = audio.Duration, Language = "en", Models = [Model] };
        int minimum = runtime.Configuration.Streaming ? 80 : 895;
        if (audio.FrameCount < minimum) throw new ArgumentException($"This Moonshine frontend needs at least {minimum} real input samples.");
        using var encoded = runtime.Encode(audio, cancellationToken);
        return DecodeEncoded(encoded["encoder"], audio.Duration, cancellationToken);
    }
    internal MoonshineModelSession Runtime => runtime;
    internal MoonshineTokenizer Tokenizer => tokenizer;
    internal double TokenRate => options.TokensPerSecond;
    internal Transcript DecodeEncoded(RecursiveAI.OneModel.Tensors.Tensor memory, TimeSpan duration, CancellationToken cancellationToken)
    {
        int maximum = Math.Max(1, (int)(duration.TotalSeconds * options.TokensPerSecond));
        // One duration-bounded ID buffer replaces a new prefix array at every step.
        var ids = new int[maximum + 1];
        ids[0] = 1;
        int used = 1;
        for (int i = 0; i < maximum; i++)
        {
            using var output = runtime.Decode(memory, ids.AsSpan(0, used), cancellationToken);
            var logits = output["logits"].Data<float>()[^32768..];
            int best = 0;
            for (int id = 0; id < logits.Length; id++)
            {
                if (!float.IsFinite(logits[id])) throw new UnsupportedAudioException("Moonshine produced nonfinite logits; provide normalized PCM samples.");
                if (logits[id] > logits[best]) best = id;
            }
            ids[used++] = best;
            if (best == 2) break;
        }
        cancellationToken.ThrowIfCancellationRequested();
        TokenTrace?.Invoke(ids[..used]);
        string text = tokenizer.Decode(ids.AsSpan(0, used), cancellationToken: cancellationToken);
        ImmutableArray<Word> words = [.. text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(t => new Word { Text = t, Language = "en" })];
        var segment = new Segment { Text = text, Start = TimeSpan.Zero, End = duration, Language = "en", Words = words };
        return new Transcript { Text = text, Duration = duration, Language = "en", Models = [Model], Words = words, Segments = text.Length == 0 ? [] : [segment] };
    }
    /// <summary>Yields the single finalized utterance segment. Limits and errors follow Transcribe; longer recordings require the VAD pipeline.</summary>
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default) => Transcribe(audio, language, cancellationToken).Segments;
    /// <summary>Refuses language detection: the fixed-English checkpoint does not infer a language from audio.</summary>
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        cancellationToken.ThrowIfCancellationRequested();
        throw new CapabilityNotSupportedException("Moonshine English checkpoints do not provide language detection. Use a multilingual recognizer for that role.");
    }
    /// <summary>Releases model graphs. Idempotent; do not dispose during a call or while a live session borrows this recognizer.</summary>
    protected override void Dispose(bool disposing) { base.Dispose(disposing); if (!disposing) return; if (disposed) return; disposed = true; runtime.Dispose(); }
}
