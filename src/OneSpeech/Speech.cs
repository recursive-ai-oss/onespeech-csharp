namespace OneSpeech;

/// <summary>Thin file-input entry points over the VAD transcription pipeline.</summary>
public static class Speech
{
    /// <summary>Aligns written text to a local audio file of at most 60 seconds, downmixed and resampled to mono 16 kHz. Downloads the tier-1 CTC model on first use. Unalignable words have null times. Throws typed capability/model/audio errors or IOException for filesystem failures.</summary>
    public static Transcript Align(string path, string text, string? language = null, string? model = null) =>
        Align(path, text, CancellationToken.None, language, model);
    /// <summary>Aligns written text with explicit cancellation, checked at audio blocks, graph calls and trellis frames. File length is limited to 60 seconds. Throws OperationCanceledException on cancellation; model, audio and capability errors remain typed.</summary>
    public static Transcript Align(string path, string text, CancellationToken cancellationToken, string? language = null, string? model = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Pipeline.SpeechServices.Align(path, text, null, language, model, cancellationToken);
    }
    /// <summary>Aligns an existing document's Words in order, preserving written text and non-timing word fields. File length is limited to 60 seconds. Downloads the tier-1 CTC model on first use. Unalignable words keep null times. Errors follow the text overload.</summary>
    public static Transcript Align(string path, Transcript transcript, string? language = null, string? model = null) =>
        Align(path, transcript, CancellationToken.None, language, model);
    /// <summary>Aligns existing written words with explicit cancellation. No word re-splitting or document-schema change is performed. File length is limited to 60 seconds; errors and cancellation follow the text overload.</summary>
    public static Transcript Align(string path, Transcript transcript, CancellationToken cancellationToken, string? language = null, string? model = null)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        return Pipeline.SpeechServices.Align(path, null, transcript, language ?? transcript.Language, model, cancellationToken);
    }

    /// <summary>Aligns text to caller-owned encoded audio without closing the stream. The stream is asynchronously spooled to a temporary seekable file (512 MiB limit), then decoded to at most 60 seconds of mono 16 kHz PCM. Model acquisition is asynchronous; in-memory inference is synchronous. Throws typed model/audio/capability errors, IOException or OperationCanceledException.</summary>
    public static ValueTask<Transcript> AlignAsync(Stream stream, string text, CancellationToken cancellationToken, string? language = null, string? model = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Pipeline.SpeechServices.AlignAsync(stream, text, null, language, model, cancellationToken);
    }
    /// <summary>Aligns existing written words to caller-owned encoded audio without re-splitting or closing the stream. Limits, spooling, errors and cancellation follow the text stream overload; non-timing word fields are retained.</summary>
    public static ValueTask<Transcript> AlignAsync(Stream stream, Transcript transcript, CancellationToken cancellationToken, string? language = null, string? model = null)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        return Pipeline.SpeechServices.AlignAsync(stream, null, transcript, language ?? transcript.Language, model, cancellationToken);
    }
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
