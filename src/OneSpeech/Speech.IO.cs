namespace OneSpeech;

public static partial class Speech
{
    /// <summary>Fetches and verifies the selected catalog model and the full companion set used by transcription/live calls, without loading inference sessions. Defaults match Transcribe; pass the live model ID to prepare Listen. Includes timing fallbacks and diarization embedders; unknown language prepares both English and Chinese embedders and the Qwen aligner. Silero is already embedded. Explicit aligner/embedder IDs pull only that role. Does not accept gates or persist tokens; typed model/capability errors and cancellation propagate. See docs/model-downloads.md for sizes and per-verb dependencies. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask PrefetchAsync(string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.PrefetchAsync(language, model, cancellationToken);

    /// <summary>Asynchronously acquires models for a local file and then performs synchronous CPU transcription. Does not spool the path or schedule CPU work on the thread pool. Defaults, companions, errors and limits follow Transcribe. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<Transcript> TranscribeAsync(string path, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.TranscribeAsync(path, language, model, cancellationToken);

    /// <summary>Yields final segments from a caller-owned encoded stream. Asynchronously spools at most 512 MiB to disk before bounded decoding; never closes the caller's stream. No whole-file diarization pass. Defaults and errors follow the path overload; cancellation or enumeration disposal deletes the spool. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static IAsyncEnumerable<Segment> TranscribeStream(Stream stream, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.TranscribeStream(stream, language, model, cancellationToken);

    /// <summary>Asynchronously acquires a recognizer for language detection on a local path. Language must be null; defaults, errors and detection behaviour follow DetectLanguage. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<string> DetectLanguageAsync(string path, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.DetectLanguageAsync(path, language, model, cancellationToken);

    /// <summary>Aligns text on a local path after asynchronous companion acquisition. At most 60 seconds; defaults, refusals and null-timing behaviour follow Align. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<Transcript> AlignAsync(string path, string text, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.AlignPathAsync(path, text, null, language, model, cancellationToken);

    /// <summary>Aligns existing written words on a local path after asynchronous model acquisition, retaining non-timing fields. Limits and errors follow Align. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<Transcript> AlignAsync(string path, Transcript transcript, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.AlignPathAsync(path, null, transcript, language ?? transcript?.Language, model, cancellationToken);
    /// <summary>Asynchronously acquires speaker weights for an encoded local path, then extracts synchronously. Consent, 60-second bound, process-wide enrollment and errors follow Enroll; no thread-pool trampoline or spool is used. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask EnrollAsync(string name, string path, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeakerServices.EnrollAsync(name, path, language, model, cancellationToken);

    /// <summary>Asynchronously acquires speaker weights for a local path, then identifies synchronously. Consent, unknown handling and errors follow Identify. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<SpeakerMatch> IdentifyAsync(string path, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeakerServices.IdentifyAsync(path, language, model, cancellationToken);

    /// <summary>Asynchronously acquires speaker weights for two local paths, then scores synchronously. Consent, sample limits and errors follow Verify. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<double> VerifyAsync(string firstPath, string secondPath, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeakerServices.VerifyAsync(firstPath, secondPath, language, model, cancellationToken);

    /// <summary>Detects language from a borrowed encoded stream after asynchronously spooling at most 512 MiB. Caller retains stream ownership. Language must be null; errors follow DetectLanguage. Companion IDs, conditional downloads and sizes are listed in docs/model-downloads.md.</summary>
    public static ValueTask<string> DetectLanguageAsync(Stream stream, string? language = null, string? model = null, CancellationToken cancellationToken = default) =>
        Pipeline.SpeechServices.DetectLanguageAsync(stream, language, model, cancellationToken);
}
