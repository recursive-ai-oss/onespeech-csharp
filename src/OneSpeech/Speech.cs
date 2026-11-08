namespace OneSpeech;

/// <summary>Thin file-input entry points over the VAD transcription pipeline.</summary>
public static class Speech
{
    /// <summary>Yields final segments from a file in bounded VAD-first chunks, without a whole-file diarization pass. Enumeration owns the decoder and models, including on early disposal. Downloads missing models on first enumeration. Optional language/model select the language and catalog ID; errors follow Transcribe.</summary>
    public static IEnumerable<Segment> TranscribeStream(string path, string? language = null, string? model = null) =>
        TranscribeStream(path, CancellationToken.None, language, model);
    /// <summary>Streams final file segments with cancellation at each chunk. Speaker assignment is not performed; no speaker IDs or word timings are invented. Errors follow Transcribe.</summary>
    public static IEnumerable<Segment> TranscribeStream(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.TranscribeStream(path, language, model, cancellationToken);
    /// <summary>Awaits mono 16 kHz PCM and yields live events. Offline families emit VAD-endpointed finals only; streaming families may emit replacing partials. Owns models and session during enumeration, not the source. Normal end flushes pending speech; early disposal does not. Downloads missing models on first enumeration. No capture, devices or speaker assignment. Errors follow Transcribe, plus UnsupportedAudioException for other PCM formats.</summary>
    public static IAsyncEnumerable<LiveEvent> Listen(IAsyncEnumerable<AudioBuffer> source, string? language = null, string? model = null) =>
        Listen(source, CancellationToken.None, language, model);
    /// <summary>Listens with explicit cancellation. Cancellation or early enumeration disposal discards pending speech and releases model resources. Input and errors follow Listen.</summary>
    public static IAsyncEnumerable<LiveEvent> Listen(IAsyncEnumerable<AudioBuffer> source, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.Listen(source, language, model, cancellationToken);
    /// <summary>Transcribes a path. Optional language/model select a language tag and catalog ID. Downloads missing tier-1 models on first use. Throws typed model errors on download or integrity failure, UnsupportedAudioException for invalid audio, or OperationCanceledException when canceled.</summary>
    public static Transcript Transcribe(string path, string? language = null, string? model = null) =>
        Transcribe(path, CancellationToken.None, language, model);
    /// <summary>Transcribes a path with explicit cancellation. Downloads missing tier-1 models on first use. Throws typed model errors on download or integrity failure, UnsupportedAudioException for invalid audio, or OperationCanceledException when canceled.</summary>
    public static Transcript Transcribe(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.Transcribe(path, language, model, cancellationToken);
    /// <summary>Transcribes a stream without owning it. Cancellation is required explicitly. Downloads missing tier-1 models on first use. Throws typed model errors on download or integrity failure, UnsupportedAudioException for invalid audio, or OperationCanceledException when canceled.</summary>
    public static ValueTask<Transcript> TranscribeAsync(Stream stream, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.TranscribeAsync(stream, language, model, cancellationToken);
    /// <summary>Detects a language from the first speech chunk of a path. Model selects a catalog ID; language must be null because detection determines it. Downloads missing tier-1 models on first use. Throws typed model errors on download or integrity failure, UnsupportedAudioException for invalid/silent audio, or CapabilityNotSupportedException if language detection is unavailable.</summary>
    public static string DetectLanguage(string path, string? language = null, string? model = null) =>
        DetectLanguage(path, CancellationToken.None, language, model);
    /// <summary>Detects a language from the first speech chunk with explicit cancellation. Model selects a catalog ID; language must be null. Downloads missing tier-1 models on first use. Throws typed model errors on download or integrity failure, UnsupportedAudioException for invalid/silent audio, or CapabilityNotSupportedException if language detection is unavailable.</summary>
    public static string DetectLanguage(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.DetectLanguage(path, language, model, cancellationToken);
}
