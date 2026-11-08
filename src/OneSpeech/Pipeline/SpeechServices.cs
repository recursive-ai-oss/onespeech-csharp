using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;
using OneSpeech.Whisper;

namespace OneSpeech.Pipeline;

internal static class SpeechServices
{
    private static async ValueTask<WhisperRecognizer> ResolveAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        var entry = model is null ? store.Catalog.GetDefault("whisper") : store.Catalog.Get(model);
        if (entry.Family != "whisper") throw new CapabilityNotSupportedException($"'{entry.Id}' is not a registered speech recognizer.");
        if (entry.ConvertedFiles.Any(f => f.Size > int.MaxValue))
            throw new CapabilityNotSupportedException($"'{entry.Id}' requires a runtime graph larger than OneModel 1.0.0's 2 GiB file limit. Select openai/whisper-large-v3-turbo until a larger-file loader is available.");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return WhisperRecognizer.Load(resolved.Directory, cancellationToken: cancellationToken);
    }

    private static async ValueTask<CampPlusEmbedder> ResolveEmbedderAsync(ModelStore store, string? language, CancellationToken cancellationToken)
    {
        var entry = store.Catalog.Get(EmbedderId(language));
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return new CampPlusEmbedder(Path.Combine(resolved.Directory, entry.ConvertedFiles.Single().Name), resolved.Info,
            dimension: IsChinese(language) ? 192 : 512, cancellationToken: cancellationToken);
    }

    internal static string EmbedderId(string? language) => IsChinese(language)
        ? "iic/speech_campplus_sv_zh-cn_16k-common" : "iic/speech_campplus_sv_en_voxceleb_16k";

    private static bool IsChinese(string? language) => language is not null &&
        (language.Equals("Chinese", StringComparison.OrdinalIgnoreCase) || language.Equals("zh", StringComparison.OrdinalIgnoreCase) || language.StartsWith("zh-", StringComparison.OrdinalIgnoreCase));

    private static async ValueTask<Transcript> DiarizeAsync(AudioFileSource source, Transcript transcript, string? language, ModelStore store, SileroVad vad, CancellationToken cancellationToken)
    {
        using var embedder = await ResolveEmbedderAsync(store, language ?? transcript.Language, cancellationToken).ConfigureAwait(false);
        var diarizer = new ClusteringDiarizer(embedder, vad);
        var turns = diarizer.DiarizeBlocks(VadChunker.Blocks(source, cancellationToken), cancellationToken);
        var merged = TranscriptDiarization.Merge(transcript, turns.Select(t => new ActiveTurn { Turn = t }), cancellationToken);
        return merged with { Models = [.. merged.Models.Concat(diarizer.Models).DistinctBy(m => (m.Id, m.Revision))] };
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
        var transcript = new VadTranscriptionPipeline(recognizer, vad).Transcribe(path, language, cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        return DiarizeAsync(source, transcript, language, store, vad, cancellationToken).AsTask().GetAwaiter().GetResult();
    }

    internal static async ValueTask<Transcript> TranscribeAsync(Stream stream, string? language, string? model, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        cancellationToken.ThrowIfCancellationRequested();
        if (!stream.CanRead) throw new ArgumentException("Input stream must be readable.", nameof(stream));
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        // Retain only encoded input so recognition can select a language before the diarization pass.
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        await using var spool = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 65536, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
        await stream.CopyToAsync(spool, cancellationToken).ConfigureAwait(false);
        await spool.FlushAsync(cancellationToken).ConfigureAwait(false);
        spool.Position = 0;
        Transcript transcript;
        using (var recognitionSource = AudioFileSource.Open(spool, true, cancellationToken))
            transcript = new VadTranscriptionPipeline(recognizer, vad).TranscribeBlocks(VadChunker.Blocks(recognitionSource, cancellationToken), language, cancellationToken);
        spool.Position = 0;
        using var source = AudioFileSource.Open(spool, true, cancellationToken);
        return await DiarizeAsync(source, transcript, language, store, vad, cancellationToken).ConfigureAwait(false);
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
