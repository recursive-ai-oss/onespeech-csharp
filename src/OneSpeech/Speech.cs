namespace OneSpeech;

/// <summary>Thin file-input entry points. Recognition is not available in this skeleton.</summary>
public static class Speech
{
    /// <summary>Transcribes a path. Optional language/model select a language tag and catalog ID. Throws FeatureUnavailableException until file recognition is installed.</summary>
    public static Transcript Transcribe(string path, string? language = null, string? model = null) =>
        Transcribe(path, CancellationToken.None, language, model);
    /// <summary>Transcribes a path with explicit cancellation. Throws FeatureUnavailableException until file recognition is installed.</summary>
    public static Transcript Transcribe(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        throw new FeatureUnavailableException("File transcription is not available yet. A file-input provider and recognizer are required.");
    /// <summary>Transcribes a stream without owning it. Cancellation is required explicitly. Throws FeatureUnavailableException until file recognition is installed.</summary>
    public static ValueTask<Transcript> TranscribeAsync(Stream stream, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        throw new FeatureUnavailableException("Stream transcription is not available yet. A file-input provider and recognizer are required.");
    /// <summary>Detects a language from a path. Optional language/model are selection hints. Throws FeatureUnavailableException until a recognizer is installed.</summary>
    public static string DetectLanguage(string path, string? language = null, string? model = null) =>
        DetectLanguage(path, CancellationToken.None, language, model);
    /// <summary>Detects a language from a path with explicit cancellation. Throws FeatureUnavailableException until a recognizer is installed.</summary>
    public static string DetectLanguage(string path, CancellationToken cancellationToken, string? language = null, string? model = null) =>
        throw new FeatureUnavailableException("Language detection is not available yet. A file-input provider and language-capable recognizer are required.");
}
