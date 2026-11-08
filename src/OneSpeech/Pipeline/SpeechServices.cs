using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;
using OneSpeech.Whisper;

namespace OneSpeech.Pipeline;

internal static class SpeechServices
{
    internal static async ValueTask<OneSpeech.Alignment.OmniAsrAligner> ResolveAlignerAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        var entry = model is null ? store.Catalog.GetDefault("omniasr-ctc") : store.Catalog.Get(model);
        if (entry.Family != "omniasr-ctc" || !entry.Capabilities.ForcedAlignment)
            throw new CapabilityNotSupportedException($"'{entry.Id}' is not a registered forced aligner.");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return new OneSpeech.Alignment.OmniAsrAligner(Path.Combine(resolved.Directory, "ctc-fp32.onemodel"), Path.Combine(resolved.Directory, "omniASR_tokenizer.model"), resolved.Info, cancellationToken: cancellationToken);
    }

    internal static Transcript Align(string path, string? text, Transcript? transcript, string? language, string? model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var source = AudioFileSource.Open(path, cancellationToken);
        var audio = ReadAlignmentAudio(source, cancellationToken);
        using var aligner = ResolveAlignerAsync(ModelStore.Default, model, cancellationToken).AsTask().GetAwaiter().GetResult();
        return transcript is null ? aligner.Align(audio, text!, language, cancellationToken) : aligner.Align(audio, transcript, language, cancellationToken);
    }

    internal static async ValueTask<Transcript> AlignAsync(Stream stream, string? text, Transcript? transcript, string? language, string? model, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        cancellationToken.ThrowIfCancellationRequested();
        if (!stream.CanRead) throw new ArgumentException("Input stream must be readable.", nameof(stream));
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        await using var spool = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 65536, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
        byte[] buffer = new byte[65536];
        long total = 0; int count;
        while ((count = await stream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) != 0)
        {
            total += count;
            if (total > 512L * 1024 * 1024) throw new CapabilityNotSupportedException("Encoded alignment streams are limited to 512 MiB. Use a local file for larger encoded inputs.");
            await spool.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
        }
        spool.Position = 0;
        using var source = AudioFileSource.Open(spool, true, cancellationToken);
        var audio = ReadAlignmentAudio(source, cancellationToken);
        using var aligner = await ResolveAlignerAsync(ModelStore.Default, model, cancellationToken).ConfigureAwait(false);
        return transcript is null ? aligner.Align(audio, text!, language, cancellationToken) : aligner.Align(audio, transcript, language, cancellationToken);
    }

    private static AudioBuffer ReadAlignmentAudio(AudioFileSource source, CancellationToken cancellationToken)
    {
        float[] samples = new float[60 * 16000];
        float[] block = new float[4096 * source.Channels];
        int length = 0, frames;
        while ((frames = source.Read(block, cancellationToken)) != 0)
        {
            if (length + frames > samples.Length) throw new CapabilityNotSupportedException("File alignment is limited to 60 seconds. Split audio with its corresponding written words.");
            for (int frame = 0; frame < frames; frame++)
            {
                double sum = 0;
                for (int channel = 0; channel < source.Channels; channel++) sum += block[frame * source.Channels + channel];
                samples[length++] = (float)(sum / source.Channels);
            }
        }
        return new AudioBuffer(samples.AsMemory(0, length), 16000);
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
        using var timing = ResolveAlignerAsync(store, null, cancellationToken).AsTask().GetAwaiter().GetResult();
        using var vad = ResolveVad(store, cancellationToken);
        using var embedder = ResolveEmbedderAsync(store, cancellationToken).AsTask().GetAwaiter().GetResult();
        var diarizer = new ClusteringDiarizer(embedder, vad);
        return new VadTranscriptionPipeline(recognizer, vad, null, null, diarizer) { TimingAligners = [timing] }.Transcribe(path, language, cancellationToken);
    }

    internal static async ValueTask<Transcript> TranscribeAsync(Stream stream, string? language, string? model, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        cancellationToken.ThrowIfCancellationRequested();
        if (!stream.CanRead) throw new ArgumentException("Input stream must be readable.", nameof(stream));
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
        using var timing = await ResolveAlignerAsync(store, null, cancellationToken).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        using var embedder = await ResolveEmbedderAsync(store, cancellationToken).ConfigureAwait(false);
        var diarizer = new ClusteringDiarizer(embedder, vad);
        return await new VadTranscriptionPipeline(recognizer, vad, null, null, diarizer) { TimingAligners = [timing] }.TranscribeAsync(stream, language, cancellationToken).ConfigureAwait(false);
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
