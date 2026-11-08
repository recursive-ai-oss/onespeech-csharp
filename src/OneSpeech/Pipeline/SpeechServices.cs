using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;
using OneSpeech.Whisper;

namespace OneSpeech.Pipeline;

internal static class SpeechServices
{
    internal static IEnumerable<Segment> TranscribeStream(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        using var recognizer = ResolveAsync(store, model, cancellationToken).AsTask().GetAwaiter().GetResult();
        using var vad = ResolveVad(store, cancellationToken);
        using var session = recognizer.CreateLiveSession(vad, language);
        using var source = AudioFileSource.Open(path, cancellationToken);
        foreach (var segment in ReadFinals(source, session, cancellationToken)) yield return segment;
    }

    internal static IEnumerable<Segment> ReadFinals(AudioFileSource source, LiveSession session, CancellationToken cancellationToken)
    {
        var interleaved = new float[4096 * source.Channels];
        var mono = new float[4096];
        int frames;
        while ((frames = source.Read(interleaved, cancellationToken)) != 0)
        {
            for (int i = 0; i < frames; i++)
            {
                double sum = 0;
                for (int channel = 0; channel < source.Channels; channel++) sum += interleaved[i * source.Channels + channel];
                mono[i] = (float)(sum / source.Channels);
            }
            foreach (var update in session.Push(new AudioBuffer(mono.AsMemory(0, frames), 16000), cancellationToken))
                if (update is FinalEvent final) yield return final.Segment;
        }
        foreach (var update in session.Complete(cancellationToken))
            if (update is FinalEvent final) yield return final.Segment;
    }

    internal static async IAsyncEnumerable<LiveEvent> Listen(IAsyncEnumerable<AudioBuffer> source, string? language, string? model, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(source);
        cancellationToken.ThrowIfCancellationRequested();
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        using var session = recognizer.CreateLiveSession(vad, language);
        await foreach (var update in session.Listen(source, cancellationToken).ConfigureAwait(false)) yield return update;
    }

    private static async ValueTask<WhisperRecognizer> ResolveAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        var entry = model is null ? store.Catalog.GetDefault("whisper") : store.Catalog.Get(model);
        if (entry.Family != "whisper") throw new CapabilityNotSupportedException($"'{entry.Id}' is not a registered speech recognizer.");
        if (entry.ConvertedFiles.Any(f => f.Size > int.MaxValue))
            throw new CapabilityNotSupportedException($"'{entry.Id}' requires a runtime graph larger than OneModel 1.0.0's 2 GiB file limit. Select openai/whisper-large-v3-turbo until a larger-file loader is available.");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return WhisperRecognizer.Load(resolved.Directory, cancellationToken: cancellationToken);
    }

    private static async ValueTask<CampPlusEmbedder> ResolveEmbedderAsync(ModelStore store, CancellationToken cancellationToken)
    {
        var entry = store.Catalog.GetDefault("speaker-campplus");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return new CampPlusEmbedder(Path.Combine(resolved.Directory, "campplus-en-fp32.onemodel"), resolved.Info, cancellationToken: cancellationToken);
    }

    private static SileroVad ResolveVad(ModelStore store, CancellationToken cancellationToken)
    {
        var entry = store.Catalog.GetDefault("silero");
        if (entry.Id != "silero/vad-v6.2.3") throw new CapabilityNotSupportedException("The registered VAD is Silero v6.2.3, embedded in the core package.");
        return new SileroVad(cancellationToken: cancellationToken);
    }

    internal static Transcript Transcribe(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        // Refuse invalid input before first-use model downloads.
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        using var recognizer = ResolveAsync(store, model, cancellationToken).AsTask().GetAwaiter().GetResult();
        using var vad = ResolveVad(store, cancellationToken);
        using var embedder = ResolveEmbedderAsync(store, cancellationToken).AsTask().GetAwaiter().GetResult();
        var diarizer = new ClusteringDiarizer(embedder, vad);
        return new VadTranscriptionPipeline(recognizer, vad, null, null, diarizer).Transcribe(path, language, cancellationToken);
    }

    internal static async ValueTask<Transcript> TranscribeAsync(Stream stream, string? language, string? model, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        cancellationToken.ThrowIfCancellationRequested();
        if (!stream.CanRead) throw new ArgumentException("Input stream must be readable.", nameof(stream));
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        using var embedder = await ResolveEmbedderAsync(store, cancellationToken).ConfigureAwait(false);
        var diarizer = new ClusteringDiarizer(embedder, vad);
        return await new VadTranscriptionPipeline(recognizer, vad, null, null, diarizer).TranscribeAsync(stream, language, cancellationToken).ConfigureAwait(false);
    }

    internal static string DetectLanguage(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        if (language is not null) throw new CapabilityNotSupportedException("Language detection determines the language from audio; omit the language hint.");
        cancellationToken.ThrowIfCancellationRequested();
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        using var recognizer = ResolveAsync(store, model, cancellationToken).AsTask().GetAwaiter().GetResult();
        using var vad = ResolveVad(store, cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        var chunker = new VadChunker(vad);
        foreach (var chunk in chunker.Read(VadChunker.Blocks(source, cancellationToken), null, cancellationToken))
            return recognizer.DetectLanguage(chunk.Audio, cancellationToken);
        throw new UnsupportedAudioException("No speech was detected; language cannot be determined.");
    }
}
