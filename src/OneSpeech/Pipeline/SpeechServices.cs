using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;
using OneSpeech.Whisper;

namespace OneSpeech.Pipeline;

internal static class SpeechServices
{
    private sealed class RecognizerLease(SpeechRecognizer recognizer, IDisposable owner, IDisposable? alignment = null) : SpeechRecognizer, IDisposable
    {
        public override ModelInfo Model => recognizer.Model;
        public override Capabilities Capabilities => recognizer.Capabilities;
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken) => recognizer.Transcribe(audio, language, cancellationToken);
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => recognizer.TranscribeStream(audio, language, cancellationToken);
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => recognizer.DetectLanguage(audio, cancellationToken);
        public void Dispose() { owner.Dispose(); alignment?.Dispose(); }
    }

    // Call after native timings and before a generic CTC/DTW fallback. The returned model is caller-owned.
    internal static async ValueTask<OneSpeech.Qwen3.Qwen3ForcedAligner?> TryResolveQwenAlignerAsync(ModelStore store, string language, CancellationToken cancellationToken)
    {
        if (!OneSpeech.Qwen3.Qwen3ForcedAligner.SupportedLanguages.Contains(language)) return null;
        var model = await store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B", cancellationToken: cancellationToken).ConfigureAwait(false);
        return OneSpeech.Qwen3.Qwen3ForcedAligner.Load(model.Directory, cancellationToken: cancellationToken);
    }

    private static async ValueTask<RecognizerLease> ResolveAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        var entry = model is null ? store.Catalog.GetDefault("whisper") : store.Catalog.Get(model);
        if (entry.Family == "qwen3-asr")
        {
            var qwenModel = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
            var recognition = OneSpeech.Qwen3.Qwen3Recognizer.Load(qwenModel.Directory, cancellationToken: cancellationToken);
            try
            {
                var alignmentModel = await store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B", cancellationToken: cancellationToken).ConfigureAwait(false);
                var alignment = OneSpeech.Qwen3.Qwen3ForcedAligner.Load(alignmentModel.Directory, cancellationToken: cancellationToken);
                recognition.WordAligner = alignment;
                return new RecognizerLease(recognition, recognition, alignment);
            }
            catch { recognition.Dispose(); throw; }
        }
        if (entry.Family != "whisper") throw new CapabilityNotSupportedException($"'{entry.Id}' is not a registered speech recognizer.");
        if (entry.ConvertedFiles.Any(f => f.Size > int.MaxValue))
            throw new CapabilityNotSupportedException($"'{entry.Id}' requires a runtime graph larger than OneModel 1.0.0's 2 GiB file limit. Select openai/whisper-large-v3-turbo until a larger-file loader is available.");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        var whisper = WhisperRecognizer.Load(resolved.Directory, cancellationToken: cancellationToken);
        return new RecognizerLease(whisper, whisper);
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
