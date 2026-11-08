using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Vad;
using OneSpeech.Whisper;

namespace OneSpeech.Pipeline;

internal static partial class SpeechServices
{
    internal static async IAsyncEnumerable<Segment> TranscribeStream(string path, string? language, string? model, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        cancellationToken.ThrowIfCancellationRequested();
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken, withQwenTiming: true).ConfigureAwait(false);
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
        language = LanguageTag.Normalize(language);
        ArgumentNullException.ThrowIfNull(source);
        cancellationToken.ThrowIfCancellationRequested();
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model ?? store.Catalog.GetDefault("moonshine").Id, cancellationToken, withQwenTiming: true).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        using var session = recognizer.CreateLiveSession(vad, language);
        using var embedder = await ResolveEmbedderAsync(store, language, cancellationToken).ConfigureAwait(false);
        using var chinese = language is null ? await ResolveEmbedderAsync(store, "zh", cancellationToken).ConfigureAwait(false) : null;
        using var diarized = new Live.OnlineDiarizationSession(session, embedder, chineseEmbedder: chinese);
        await foreach (var update in diarized.Listen(source, cancellationToken).ConfigureAwait(false)) yield return update;
    }

    internal static async ValueTask<OneSpeech.Alignment.OmniAsrAligner> ResolveAlignerAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        var entry = model is null ? store.Catalog.GetDefault("omniasr-ctc") : store.Catalog.Get(model);
        if (entry.Family != "omniasr-ctc" || !entry.Capabilities.ForcedAlignment)
            throw new CapabilityNotSupportedException($"'{entry.Id}' is not a registered forced aligner.");
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return new OneSpeech.Alignment.OmniAsrAligner(Path.Combine(resolved.Directory, "ctc-fp32.onemodel"), Path.Combine(resolved.Directory, "omniASR_tokenizer.model"), resolved.Info, cancellationToken: cancellationToken);
    }

    private sealed class AlignerLease(Aligner aligner) : Aligner, IDisposable
    {
        public override ModelInfo Model => aligner.Model;
        public override Capabilities Capabilities => aligner.Capabilities;
        public override Transcript Align(AudioBuffer audio, string text, string? language = null, CancellationToken cancellationToken = default) => aligner.Align(audio, text, language, cancellationToken);
        public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language = null, CancellationToken cancellationToken = default) => aligner.Align(audio, transcript, language, cancellationToken);
        protected override void Dispose(bool disposing) { try { if (disposing) ((IDisposable)aligner).Dispose(); } finally { base.Dispose(disposing); } }
    }

    private static async ValueTask<AlignerLease> ResolvePublicAlignerAsync(ModelStore store, string? model, CancellationToken cancellationToken)
    {
        if (model is null) return new(new OrderedTimingAligner(store, realign: true));
        var entry = store.Catalog.Get(model);
        if (entry.Family == "qwen3-aligner")
        {
            var resolved = await store.PullAsync(model, cancellationToken: cancellationToken).ConfigureAwait(false);
            return new(OneSpeech.Qwen3.Qwen3ForcedAligner.Load(resolved.Directory, cancellationToken: cancellationToken));
        }
        return new(await ResolveAlignerAsync(store, model, cancellationToken).ConfigureAwait(false));
    }

    internal static Transcript Align(string path, string? text, Transcript? transcript, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        cancellationToken.ThrowIfCancellationRequested();
        using var source = AudioFileSource.Open(path, cancellationToken);
        var audio = ReadAlignmentAudio(source, cancellationToken);
        using var aligner = ResolvePublicAlignerAsync(ModelStore.Default, model, cancellationToken).AsTask().GetAwaiter().GetResult();
        return transcript is null ? aligner.Align(audio, text!, language, cancellationToken) : aligner.Align(audio, transcript, language, cancellationToken);
    }

    internal static async ValueTask<Transcript> AlignAsync(Stream stream, string? text, Transcript? transcript, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        await using var spool = await SpoolAsync(stream, cancellationToken).ConfigureAwait(false);
        return await AlignPathAsync(spool.Name, text, transcript, language, model, cancellationToken).ConfigureAwait(false);
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

    private static async ValueTask<RecognizerLease> ResolveAsync(ModelStore store, string? model, CancellationToken cancellationToken, bool withQwenTiming = false)
    {
        if (model is not null && model.StartsWith("moonshine-ai/", StringComparison.Ordinal)) _ = Moonshine.MoonshineProvenance.Get(model);
        var entry = model is null ? store.Catalog.GetRoleDefault("transcription") : store.Catalog.Get(model);
        if (entry.Family == "nemotron-asr")
        {
            var resolvedNemotron = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
            var recognizer = FastConformer.NemotronAsrRecognizer.Load(resolvedNemotron, cancellationToken: cancellationToken);
            return new RecognizerLease(recognizer, recognizer);
        }
        if (entry.Family == "parakeet-tdt")
        {
            var resolvedParakeet = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
            var recognizer = FastConformer.ParakeetRecognizer.Load(resolvedParakeet, cancellationToken: cancellationToken);
            return new RecognizerLease(recognizer, recognizer);
        }
        if (entry.Family == "qwen3-asr")
        {
            var qwenModel = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
            var recognition = OneSpeech.Qwen3.Qwen3Recognizer.Load(qwenModel.Directory, cancellationToken: cancellationToken);
            if (!withQwenTiming) return new RecognizerLease(recognition, recognition);
            // Live utterances cannot replay a file after releasing recognition state.
            try
            {
                var aligned = await store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B", cancellationToken: cancellationToken).ConfigureAwait(false);
                var alignment = OneSpeech.Qwen3.Qwen3ForcedAligner.Load(aligned.Directory, cancellationToken: cancellationToken);
                recognition.WordAligner = alignment;
                return new RecognizerLease(recognition, recognition, alignment);
            }
            catch { recognition.Dispose(); throw; }
        }
        if (entry.Family is not ("whisper" or "moonshine")) throw new CapabilityNotSupportedException($"Model {entry.Id} is not a registered speech recognizer.");
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
        public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default) => recognizer.Transcribe(audio, language, cancellationToken);
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default) => recognizer.TranscribeStream(audio, language, cancellationToken);
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken = default) => recognizer.DetectLanguage(audio, cancellationToken);
        public override LiveSession CreateLiveSession(VoiceActivityDetector detector, string? language = null) => recognizer.CreateLiveSession(detector, language);
        private bool disposed;
        protected override void Dispose(bool disposing) { base.Dispose(disposing); if (!disposing) return; if (disposed) return; disposed = true; owner.Dispose(); alignment?.Dispose(); }
    }

    private static async ValueTask<CampPlusEmbedder> ResolveEmbedderAsync(ModelStore store, string? language, CancellationToken cancellationToken)
    {
        var entry = store.Catalog.Get(EmbedderId(language));
        var resolved = await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return new CampPlusEmbedder(Path.Combine(resolved.Directory, entry.ConvertedFiles.Single().Name), resolved.Info,
            dimension: IsChinese(language) ? 192 : 512, cancellationToken: cancellationToken);
    }

    internal static string EmbedderId(string? language) => IsChinese(LanguageTag.Normalize(language))
        ? "iic/speech_campplus_sv_zh-cn_16k-common" : "iic/speech_campplus_sv_en_voxceleb_16k";

    private static bool IsChinese(string? language) => language is not null &&
        (language.Equals("Chinese", StringComparison.OrdinalIgnoreCase) || language.Equals("zh", StringComparison.OrdinalIgnoreCase) || language.StartsWith("zh-", StringComparison.OrdinalIgnoreCase));

    internal static SileroVadOptions DiarizationActivity => SileroVadOptions.Diarization;

    private static async ValueTask<Transcript> DiarizeAsync(AudioFileSource source, Transcript transcript, string? language, ModelStore store, CancellationToken cancellationToken)
    {
        using var embedder = await ResolveEmbedderAsync(store, language ?? transcript.Language, cancellationToken).ConfigureAwait(false);
        // Activity coverage for diarization is fitted separately; ASR chunking keeps its original VAD.
        using var vad = ResolveVad(store, cancellationToken, DiarizationActivity);
        var diarizer = new ClusteringDiarizer(embedder, vad)
        {
            IdentifyCluster = (vector, token) => SpeakerServices.Match(embedder.Model.Id, vector, token)
        };
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
        language = LanguageTag.Normalize(language);
        cancellationToken.ThrowIfCancellationRequested();
        // Refuse invalid input before first-use model downloads.
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        GraphTiming.BeginPhase();
        using var recognizer = ResolveAsync(store, model, cancellationToken).AsTask().GetAwaiter().GetResult();
        GraphTiming.EndPhase(0);
        using var timing = recognizer.Capabilities.NativeTimestamps ? null : new OrderedTimingAligner(store);
        using var vad = ResolveVad(store, cancellationToken);
        var deferred = timing is null ? null : new DeferredTiming();
        GraphTiming.BeginPhase();
        var transcript = new VadTranscriptionPipeline(recognizer, vad)
        {
            RecognizedChunk = deferred is null ? null : deferred.Add
        }.Transcribe(path, language, cancellationToken);
        GraphTiming.EndPhase(1);
        recognizer.Dispose();
        if (deferred is not null && timing is not null)
        {
            using var replay = AudioFileSource.Open(path, cancellationToken);
            using var replayVad = ResolveVad(store, cancellationToken);
            GraphTiming.BeginPhase();
            transcript = deferred.Apply(VadChunker.Blocks(replay, cancellationToken), replayVad, transcript, timing, cancellationToken);
            GraphTiming.EndPhase(2);
        }
        timing?.Dispose();
        using var source = AudioFileSource.Open(path, cancellationToken);
        GraphTiming.BeginPhase();
        var diarized = DiarizeAsync(source, transcript, language, store, cancellationToken).AsTask().GetAwaiter().GetResult();
        GraphTiming.EndPhase(3);
        return diarized;
    }

    internal static async ValueTask<Transcript> TranscribeAsync(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        cancellationToken.ThrowIfCancellationRequested();
        // Refuse invalid input before first-use model downloads.
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        await PrefetchAsync(language, model, cancellationToken).ConfigureAwait(false);
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
        using var timing = recognizer.Capabilities.NativeTimestamps ? null : new OrderedTimingAligner(store);
        using var vad = ResolveVad(store, cancellationToken);
        var deferred = timing is null ? null : new DeferredTiming();
        var transcript = new VadTranscriptionPipeline(recognizer, vad)
        {
            RecognizedChunk = deferred is null ? null : deferred.Add
        }.Transcribe(path, language, cancellationToken);
        recognizer.Dispose();
        if (deferred is not null && timing is not null)
        {
            using var replay = AudioFileSource.Open(path, cancellationToken);
            using var replayVad = ResolveVad(store, cancellationToken);
            transcript = deferred.Apply(VadChunker.Blocks(replay, cancellationToken), replayVad, transcript, timing, cancellationToken);
        }
        timing?.Dispose();
        using var source = AudioFileSource.Open(path, cancellationToken);
        return await DiarizeAsync(source, transcript, language, store, cancellationToken).ConfigureAwait(false);
    }

    internal static async ValueTask<Transcript> TranscribeAsync(Stream stream, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        await using var spool = await SpoolAsync(stream, cancellationToken).ConfigureAwait(false);
        return await TranscribeAsync(spool.Name, language, model, cancellationToken).ConfigureAwait(false);
    }

    internal static string DetectLanguage(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
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
    internal static async ValueTask<string> DetectLanguageAsync(string path, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        if (language is not null) throw new CapabilityNotSupportedException("Language detection determines the language from audio; omit the language hint.");
        cancellationToken.ThrowIfCancellationRequested();
        using (AudioFileSource.Open(path, cancellationToken)) { }
        var store = ModelStore.Default;
        using var recognizer = await ResolveAsync(store, model, cancellationToken).ConfigureAwait(false);
        using var vad = ResolveVad(store, cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        var chunker = new VadChunker(vad);
        foreach (var chunk in chunker.Read(VadChunker.Blocks(source, cancellationToken), null, cancellationToken))
            return recognizer.DetectLanguage(chunk.Audio, cancellationToken);
        throw new UnsupportedAudioException("No speech was detected; language cannot be determined.");
    }
}
