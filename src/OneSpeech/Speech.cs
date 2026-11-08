namespace OneSpeech;

/// <summary>Thin file-input entry points over the VAD transcription pipeline.</summary>
public static class Speech
{
    /// <summary>Transcribes a path. Optional language/model select a language tag and catalog ID. Throws FeatureUnavailableException if no recognizer is configured, UnsupportedAudioException for invalid audio, or OperationCanceledException when canceled.</summary>
    public static Transcript Transcribe(string path, string? language = null, string? model = null) =>
        Transcribe(path, CancellationToken.None, language, model);
    /// <summary>Transcribes a path with explicit cancellation. Throws FeatureUnavailableException if no recognizer is configured, UnsupportedAudioException for invalid audio, or OperationCanceledException when canceled.</summary>
    public static Transcript Transcribe(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.Transcribe(path, language, model, cancellationToken);
    /// <summary>Transcribes a stream without owning it. Cancellation is required explicitly. Throws FeatureUnavailableException if no recognizer is configured, UnsupportedAudioException for invalid audio, or OperationCanceledException when canceled.</summary>
    public static ValueTask<Transcript> TranscribeAsync(Stream stream, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.TranscribeAsync(stream, language, model, cancellationToken);
    /// <summary>Detects a language from the first speech chunk of a path. Model selects a catalog ID; language must be null because detection determines it. Throws FeatureUnavailableException if no recognizer is configured, UnsupportedAudioException for invalid/silent audio, or CapabilityNotSupportedException if language detection is unavailable.</summary>
    public static string DetectLanguage(string path, string? language = null, string? model = null) =>
        DetectLanguage(path, CancellationToken.None, language, model);
    /// <summary>Detects a language from the first speech chunk with explicit cancellation. Model selects a catalog ID; language must be null. Throws FeatureUnavailableException if no recognizer is configured, UnsupportedAudioException for invalid/silent audio, or CapabilityNotSupportedException if language detection is unavailable.</summary>
    public static string DetectLanguage(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        Pipeline.SpeechServices.DetectLanguage(path, language, model, cancellationToken);
}
