using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Text.Json;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Decoding;
using OneSpeech.Vad;
using RecursiveAI.OneModel.Graph;

namespace OneSpeech.FastConformer;

/// <summary>Nemotron 3.5 ASR fp32 CPU options. No translation or phrase biasing is supported.</summary>
public sealed record NemotronAsrOptions
{
    /// <summary>Chunk duration in milliseconds: 80, 320 (default), 560, or 1120. Offline inference uses the same attention context.</summary>
    public int ChunkMilliseconds { get; init; } = 320;
    /// <summary>Intra-op CPU thread limit, 1 by default, nonnegative; zero selects the runtime default.</summary>
    public int Threads { get; init; } = 1;
}

/// <summary>Publisher-only tier-2 Nemotron 3.5 ASR, with caller-owned cache state and genuine live partials. Not thread-safe; borrowed sessions must be disposed before the recognizer.</summary>
public sealed partial class NemotronAsrRecognizer : SpeechRecognizer, IDisposable
{
    internal Session First { get; }
    internal Session Last { get; }
    internal Session Tail { get; }
    internal Session Prompt { get; }
    internal Session Predictor { get; }
    internal Session Joint { get; }
    internal float[] Embeddings { get; }
    internal ParakeetVocabulary Vocabulary { get; }
    internal int ChunkFrames { get; }
    private readonly Dictionary<string, int> prompts;
    private bool disposed;
    internal const string Identifier = "nvidia/nemotron-3.5-asr-streaming-0.6b";
    internal const string Revision = "ea30d66debe3740a08b573244286791d423d6b3e";
    internal static readonly ImmutableArray<string> SupportedLanguages = ["en", "es", "fr", "it", "pt", "nl", "de", "tr", "ru", "ar", "hi", "ja", "ko", "vi", "uk", "pl", "sv", "cs", "nb", "da", "bg", "fi", "hr", "sk", "zh", "hu", "ro", "et", "en-US", "en-GB", "es-US", "es-ES", "fr-FR", "fr-CA", "it-IT", "pt-BR", "pt-PT", "nl-NL", "de-DE", "tr-TR", "ru-RU", "ar-AR", "hi-IN", "ja-JP", "ko-KR", "vi-VN", "uk-UA", "pl-PL", "sv-SE", "cs-CZ", "nb-NO", "da-DK", "bg-BG", "fi-FI", "hr-HR", "sk-SK", "zh-CN", "hu-HU", "ro-RO", "et-EE"];
    private NemotronAsrRecognizer(ResolvedModel resolved, NemotronAsrOptions options, CancellationToken ct)
    {
        if (resolved.Info.Id != Identifier || resolved.Info.Revision != Revision || resolved.Info.Tier != ModelTier.Tier2 || resolved.Info.License != "OpenMDW-1.1" || !resolved.Info.Attribution.Contains("OpenMDW License Agreement, version 1.1", StringComparison.Ordinal)) throw new ModelIntegrityException("Expected the pinned publisher Nemotron 3.5 ASR checkpoint.");
        if (options.ChunkMilliseconds is not (80 or 320 or 560 or 1120) || options.Threads < 0) throw new ArgumentOutOfRangeException(nameof(options));
        var official = ModelCatalog.Default.Get(Identifier);
        if (resolved.Info.Source != official.Source || !resolved.Info.Attribution.Contains(official.Attribution, StringComparison.Ordinal) || official.Files.Any(file => !resolved.Info.Files.Any(actual => actual.Name == file.Name && actual.Sha256 == file.Sha256)))
            throw new ModelIntegrityException("Resolve this checkpoint through ModelStore to retain its publisher hashes and complete OpenMDW notices.");
        Model = resolved.Info; ChunkFrames = options.ChunkMilliseconds / 80;
        Vocabulary = new(Path.Combine(resolved.Directory, "tokenizer.json"), 13087);
        prompts = ReadPrompts(Path.Combine(resolved.Directory, "processor_config.json"));
        using (var file = File.OpenRead(Path.Combine(resolved.Directory, "embedding.f32")))
        {
            if (file.Length != 13088L * 640 * 4) throw new ModelIntegrityException("Invalid Nemotron embedding size.");
            Embeddings = new float[13088 * 640]; byte[] buffer = new byte[2560];
            for (int row = 0; row < 13088; row++) { ct.ThrowIfCancellationRequested(); file.ReadExactly(buffer); for (int col = 0; col < 640; col++) Embeddings[row * 640 + col] = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(col * 4)); }
        }
        var sessions = new List<Session>();
        Session Load(string name) { var session = Session.Load(Path.Combine(resolved.Directory, name + ".onemodel"), new ModelOptions { IntraOpThreads = options.Threads, MaxCachedPlans = 4 }, cancellationToken: ct); sessions.Add(session); return session; }
        try { First = Load("encoder-step0"); Last = Load("encoder-step1"); Tail = Load("encoder-tail0"); Prompt = Load("prompt"); Predictor = Load("predictor"); Joint = Load("joint"); }
        catch { foreach (var session in sessions) session.Dispose(); throw; }
    }
    /// <summary>Loads verified locally converted publisher files. Defaults to 320 ms chunks and runtime-selected CPU threads. Throws ModelIntegrityException for wrong provenance/artifacts, ArgumentOutOfRangeException for invalid options, IOException for missing files, or OperationCanceledException.</summary>
    public static NemotronAsrRecognizer Load(ResolvedModel model, NemotronAsrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model); cancellationToken.ThrowIfCancellationRequested(); return new(model, options ?? new(), cancellationToken);
    }
    /// <inheritdoc />
    public override ModelInfo Model { get; }
    /// <summary>Transcription-ready and broad-coverage base languages, native RNN-T frame timing, true streaming. Adaptation-only locales are not advertised.</summary>
    public override Capabilities Capabilities => new() { Languages = SupportedLanguages, Streaming = true, NativeTimestamps = true };
    internal static Dictionary<string, int> ReadPrompts(string config)
    {
        if (new FileInfo(config).Length > 1048576) throw new ModelIntegrityException("Processor configuration exceeds 1 MiB.");
        var prompts = new Dictionary<string, int>(StringComparer.Ordinal);
        try
        {
            using var json = JsonDocument.Parse(File.ReadAllBytes(config), new JsonDocumentOptions { MaxDepth = 16 });
            foreach (var item in json.RootElement.GetProperty("prompt_dictionary").EnumerateObject())
            {
                int id = item.Value.GetInt32(); if (id is < 0 or >= 128 || prompts.Count >= 256) throw new ModelIntegrityException("Invalid language prompt dictionary.");
                prompts.Add(item.Name, id);
            }
            if (!prompts.TryGetValue("auto", out int auto) || auto != 101 || !prompts.TryGetValue("en-US", out int english) || english != 0) throw new ModelIntegrityException("Expected the pinned language prompt dictionary.");
            foreach (string language in SupportedLanguages)
            {
                if (prompts.ContainsKey(language)) continue;
                string? locale = SupportedLanguages.FirstOrDefault(candidate => candidate.StartsWith(language + "-", StringComparison.Ordinal) && prompts.ContainsKey(candidate));
                if (locale is null) throw new ModelIntegrityException("Missing declared language prompt: " + language);
                prompts.Add(language, prompts[locale]);
            }
        }
        catch (Exception error) when (error is JsonException or KeyNotFoundException or FormatException or InvalidOperationException or ArgumentException)
        { throw new ModelIntegrityException("Invalid Nemotron processor configuration.", error); }
        return prompts;
    }
    internal int PromptId(string? language)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (language is null) return 101;
        if (!SupportedLanguages.Contains(language) || !prompts.TryGetValue(language, out int id)) throw new CapabilityNotSupportedException("Unsupported Nemotron language '" + language + "'. Use a declared language code or omit it for auto detection.");
        return id;
    }
    /// <inheritdoc />
    public override LiveSession CreateLiveSession(VoiceActivityDetector detector, string? language = null)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(detector); int id = PromptId(language);
        return new Live.UtteranceLiveSession(this, detector, new NemotronAsrState(this, id, CancellationToken.None), language);
    }
    /// <summary>Transcribes mono 16 kHz finite PCM. Inputs over 30 s use bounded VAD chunks. Shorter than one mel hop returns empty. Language null selects the auto prompt. Throws ArgumentException for invalid PCM, CapabilityNotSupportedException, ObjectDisposedException or OperationCanceledException.</summary>
    public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(audio); int id = PromptId(language); cancellationToken.ThrowIfCancellationRequested();
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Nemotron requires mono 16000 Hz PCM.");
        foreach (float value in audio.Samples.Span) if (!float.IsFinite(value)) throw new ArgumentException("PCM must be finite.");
        if (audio.FrameCount > 480000) { using var vad = new SileroVad(cancellationToken: cancellationToken); return new Pipeline.VadTranscriptionPipeline(this, vad).Transcribe(audio, language, cancellationToken); }
        using var state = new NemotronAsrState(this, id, cancellationToken, padFinalChunk: false); state.Push(audio.Samples.Span, true, cancellationToken);
        var segment = Snapshot(state.Tokens, TimeSpan.Zero, audio.Duration, language ?? state.DetectedLanguage, cancellationToken);
        return new Transcript { Text = segment.Text, Duration = audio.Duration, Language = segment.Language, Words = segment.Words, Segments = [segment], Models = [Model] };
    }
    internal Segment Snapshot(ImmutableArray<FrameToken> tokens, TimeSpan start, TimeSpan end, string? language, CancellationToken ct)
    {
        var words = NativeWordTiming.Map(tokens.AsSpan(), Vocabulary.Words(tokens.AsSpan()), TimeSpan.FromMilliseconds(80), cancellationToken: ct);
        return new Segment { Start = start, End = end, Text = Vocabulary.Decode(tokens.Select(x => x.Id).ToArray()), Language = language,
            Words = [.. words.Select(w => w.Start + start >= end
                ? w with { Start = null, End = null, TimingMethod = TimingMethod.None, Language = language }
                : w with { Start = w.Start + start, End = w.End + start > end ? end : w.End + start, Language = language })] };
    }
    /// <inheritdoc />
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        language = LanguageTag.Require(language, Capabilities, Model.Id);
        ArgumentNullException.ThrowIfNull(audio); _ = PromptId(language); cancellationToken.ThrowIfCancellationRequested();
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Nemotron requires mono 16000 Hz PCM.");
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        using var session = CreateLiveSession(vad, language);
        for (int offset = 0; offset < audio.FrameCount; offset += 1600)
            foreach (var update in session.Push(new AudioBuffer(audio.Samples.Slice(offset, Math.Min(1600, audio.FrameCount - offset)), audio.SampleRate, audio.Channels), cancellationToken))
                if (update is FinalEvent final) yield return final.Segment;
        foreach (var update in session.Complete(cancellationToken)) if (update is FinalEvent final) yield return final.Segment;
    }
    /// <summary>Runs automatic-prompt transcription and returns its emitted language tag. No confidence is invented. Throws UnsupportedAudioException if no tag was emitted; other errors follow Transcribe.</summary>
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken = default) => Transcribe(audio, null, cancellationToken).Language ?? throw new UnsupportedAudioException("No language tag was emitted; provide a longer speech sample or an explicit language.");
    /// <summary>Releases model graph resources. Idempotent; dispose borrowed live sessions first.</summary>
    protected override void Dispose(bool disposing) { base.Dispose(disposing); if (!disposing) return; if (disposed) return; disposed = true; First.Dispose(); Last.Dispose(); Tail.Dispose(); Prompt.Dispose(); Predictor.Dispose(); Joint.Dispose(); }
}
