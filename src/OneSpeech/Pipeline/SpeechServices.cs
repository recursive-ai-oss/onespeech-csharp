using OneSpeech.Audio;
using OneSpeech.Vad;

namespace OneSpeech.Pipeline;

// A borrowed recognizer is supplied by the model-loading integration, never discovered by reflection.
internal static class SpeechServices
{
    internal static Func<string?, CancellationToken, SpeechRecognizer>? RecognizerResolver { get; set; }

    private static SpeechRecognizer Resolve(string? model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return RecognizerResolver?.Invoke(model, cancellationToken)
            ?? throw new FeatureUnavailableException("No speech recognizer is installed. Configure the model resolver with a supported recognition model.");
    }

    internal static Transcript Transcribe(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        var recognizer = Resolve(model, cancellationToken);
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        return new VadTranscriptionPipeline(recognizer, vad).Transcribe(path, language, cancellationToken);
    }

    internal static async ValueTask<Transcript> TranscribeAsync(Stream stream, string? language, string? model, CancellationToken cancellationToken)
    {
        var recognizer = Resolve(model, cancellationToken);
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        return await new VadTranscriptionPipeline(recognizer, vad).TranscribeAsync(stream, language, cancellationToken).ConfigureAwait(false);
    }

    internal static string DetectLanguage(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        var recognizer = Resolve(model, cancellationToken);
        if (language is not null) throw new CapabilityNotSupportedException("Language detection determines the language from audio; omit the language hint.");
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        var chunker = new VadChunker(vad);
        foreach (var chunk in chunker.Read(VadChunker.Blocks(source, cancellationToken), null, cancellationToken))
            return recognizer.DetectLanguage(chunk.Audio, cancellationToken);
        throw new UnsupportedAudioException("No speech was detected; language cannot be determined.");
    }
}
