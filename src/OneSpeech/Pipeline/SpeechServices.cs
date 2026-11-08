using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;
using OneSpeech.Whisper;

namespace OneSpeech.Pipeline;

internal static class SpeechServices
{
    internal static async IAsyncEnumerable<Segment> TranscribeStream(string path, string? language, string? model, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
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
        using var recognizer = await ResolveAsync(store, model ?? store.Catalog.GetDefault("moonshine").Id, cancellationToken).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        using var session = recognizer.CreateLiveSession(vad, language);
        await foreach (var update in session.Listen(source, cancellationToken).ConfigureAwait(false)) yield return update;
    }

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

    // Call after native timings and before a generic CTC/DTW fallback. The returned model is caller-owned.
    internal static async ValueTask<OneSpeech.Qwen3.Qwen3ForcedAligner?> TryResolveQwenAlignerAsync(ModelStore store, string language, CancellationToken cancellationToken)
    {
        if (!OneSpeech.Qwen3.Qwen3ForcedAligner.SupportedLanguages.Contains(language)) return null;
        var model = await store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B", cancellationToken: cancellationToken).ConfigureAwait(false);
        return OneSpeech.Qwen3.Qwen3ForcedAligner.Load(model.Directory, cancellationToken: cancellationToken);
    }

    private static async ValueTask<RecognizerLease> ResolveAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        if (model is not null && model.StartsWith("moonshine-ai/", StringComparison.Ordinal)) _ = Moonshine.MoonshineProvenance.Get(model);
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
        if (entry.Family is not ("whisper" or "moonshine")) throw new CapabilityNotSupportedException($"Model {entry.Id} is not a registered speech recognizer.");
        if (entry.ConvertedFiles.Any(f => f.Size > int.MaxValue))
            throw new CapabilityNotSupportedException($"'{entry.Id}' requires a runtime graph larger than OneModel 1.0.0's 2 GiB file limit. Select openai/whisper-large-v3-turbo until a larger-file loader is available.");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        if (entry.Family == "moonshine")
        {
            var recognizer = Moonshine.MoonshineRecognizer.Load(resolved.Directory, cancellationToken: cancellationToken);
            return new RecognizerLease(recognizer, recognizer);
        }
        var whisper = WhisperRecognizer.Load(resolved.Directory, cancellationToken: cancellationToken);
        return new RecognizerLease(whisper, whisper);
    }

    private sealed class RecognizerLease(SpeechRecognizer recognizer, IDisposable owner, IDisposable? alignment = null) : SpeechRecognizer, IDisposable
    {
        public override ModelInfo Model => recognizer.Model;
        public override Capabilities Capabilities => recognizer.Capabilities;
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken) => recognizer.Transcribe(audio, language, cancellationToken);
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => recognizer.TranscribeStream(audio, language, cancellationToken);
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => recognizer.DetectLanguage(audio, cancellationToken);
        public override LiveSession CreateLiveSession(SileroVad detector, string? language = null) => recognizer.CreateLiveSession(detector, language);
        public void Dispose() { owner.Dispose(); alignment?.Dispose(); }
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

    internal static SileroVadOptions DiarizationActivity => new()
    {
        Threshold = 0.2f, NegativeThreshold = 0.2f - 0.15f,
        MinimumSpeech = TimeSpan.FromMilliseconds(250), MinimumSilence = TimeSpan.FromMilliseconds(250),
        SpeechPadding = TimeSpan.FromMilliseconds(150)
    };

    internal static CosineResegmenter DiarizationResegmentation => new(0.4, 2);

    private static async ValueTask<Transcript> DiarizeAsync(AudioFileSource source, Transcript transcript, string? language, ModelStore store, CancellationToken cancellationToken)
    {
        using var embedder = await ResolveEmbedderAsync(store, language ?? transcript.Language, cancellationToken).ConfigureAwait(false);
        // Activity coverage for diarization is fitted separately; ASR chunking keeps its original VAD.
        using var vad = ResolveVad(store, cancellationToken, DiarizationActivity);
        var diarizer = new ClusteringDiarizer(embedder, vad, resegmenter: DiarizationResegmentation);
        var turns = diarizer.DiarizeBlocks(VadChunker.Blocks(source, cancellationToken), cancellationToken);
        var merged = TranscriptDiarization.Merge(transcript, turns.Select(t => new ActiveTurn { Turn = t }), cancellationToken);
        return merged with { Models = [.. merged.Models.Concat(diarizer.Models).DistinctBy(m => (m.Id, m.Revision))] };
    }

    private static SileroVad ResolveVad(ModelStore store, CancellationToken cancellationToken, SileroVadOptions? options = null)
    {
        var entry = store.Catalog.GetDefault("silero");
        if (entry.Id != "silero/vad-v6.2.3") throw new CapabilityNotSupportedException("The registered VAD is Silero v6.2.3, embedded in the core package.");
        return new SileroVad(options, cancellationToken);
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
        var transcript = new VadTranscriptionPipeline(recognizer, vad) { TimingAligners = [timing] }.Transcribe(path, language, cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        return DiarizeAsync(source, transcript, language, store, cancellationToken).AsTask().GetAwaiter().GetResult();
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
        // Retain only encoded input so recognition can select a language before the diarization pass.
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        await using var spool = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 65536, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
        await stream.CopyToAsync(spool, cancellationToken).ConfigureAwait(false);
        await spool.FlushAsync(cancellationToken).ConfigureAwait(false);
        spool.Position = 0;
        Transcript transcript;
        using (var recognitionSource = AudioFileSource.Open(spool, true, cancellationToken))
            transcript = new VadTranscriptionPipeline(recognizer, vad) { TimingAligners = [timing] }.TranscribeBlocks(VadChunker.Blocks(recognitionSource, cancellationToken), language, cancellationToken);
        spool.Position = 0;
        using var source = AudioFileSource.Open(spool, true, cancellationToken);
        return await DiarizeAsync(source, transcript, language, store, cancellationToken).ConfigureAwait(false);
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
