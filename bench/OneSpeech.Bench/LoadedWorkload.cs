using System.Collections.Immutable;
using System.Diagnostics;
using OneSpeech;
using OneSpeech.Alignment;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;
using OneSpeech.Pipeline;
using OneSpeech.Qwen3;
using OneSpeech.Vad;
using OneSpeech.Whisper;

// The file helper's stages, with borrowed models retained across calls. In particular,
// the same deferred timing replay is used; this does not replace it with chunk timing.
internal sealed class LoadedWorkload : IDisposable
{
    private readonly List<IDisposable> owners = [];
    private SpeechRecognizer? recognizer;
    private Aligner? aligner;
    private SileroVad? recognitionVad;
    private SileroVad? replayVad;
    private ClusteringDiarizer? diarizer;
    private volatile string phase = "idle";
    internal string Phase => phase;
    internal double PrepareSeconds { get; private set; }
    internal double LoadSeconds { get; private set; }
    private string row = "";

    internal static async Task<LoadedWorkload> LoadAsync(string row, string id, int threads)
    {
        var result = new LoadedWorkload { row = row };
        try
        {
            var assets = new Dictionary<string, ResolvedModel>(StringComparer.Ordinal);
            var clock = Stopwatch.StartNew();
            await Pull(id);
            if (row is not ("align" or "diarize"))
            {
                await Pull("Qwen/Qwen3-ForcedAligner-0.6B");
                await Pull("facebook/omniASR-CTC-300M");
            }
            if (row != "align") await Pull("iic/speech_campplus_sv_en_voxceleb_16k");
            result.PrepareSeconds = clock.Elapsed.TotalSeconds;
            clock.Restart();
            if (row == "align") result.aligner = Omni();
            else
            {
                var model = assets["iic/speech_campplus_sv_en_voxceleb_16k"];
                var embedder = result.Own(new CampPlusEmbedder(Path.Combine(model.Directory, "campplus-en-fp32.onemodel"), model.Info, threads: threads));
                var activity = result.Own(new SileroVad(SileroVadOptions.Diarization));
                result.diarizer = new ClusteringDiarizer(embedder, activity)
                {
                    IdentifyCluster = (vector, token) => SpeakerServices.Match(embedder.Model.Id, vector, token)
                };
                if (row != "diarize")
                {
                    result.recognizer = id.StartsWith("Qwen/", StringComparison.Ordinal)
                        ? result.Own(Qwen3Recognizer.Load(assets[id].Directory))
                        : result.Own(WhisperRecognizer.Load(assets[id].Directory));
                    result.recognitionVad = result.Own(new SileroVad());
                    result.replayVad = result.Own(new SileroVad());
                    var qwen = result.Own(Qwen3ForcedAligner.Load(assets["Qwen/Qwen3-ForcedAligner-0.6B"].Directory));
                    result.aligner = new BorrowedTiming(qwen, () => Omni(false));
                }
            }
            result.LoadSeconds = clock.Elapsed.TotalSeconds;
            return result;

            async Task Pull(string modelId) => assets[modelId] = await ModelStore.Default.PullAsync(modelId);
            OmniAsrAligner Omni(bool owned = true)
            {
                var model = assets["facebook/omniASR-CTC-300M"];
                var timer = new OmniAsrAligner(Path.Combine(model.Directory, "ctc-fp32.onemodel"), Path.Combine(model.Directory, "omniASR_tokenizer.model"), model.Info, threads);
                return owned ? result.Own(timer) : timer;
            }
        }
        catch { result.Dispose(); throw; }
    }

    private T Own<T>(T model) where T : IDisposable { owners.Add(model); return model; }

    internal Transcript Run(string file, string text, double durationSeconds)
    {
        if (row == "align")
        {
            phase = "alignment";
            var samples = Fixtures.ReadMono(file);
            var result = aligner!.Align(new AudioBuffer(samples, 16000), text, "en", default);
            phase = "complete";
            return result;
        }
        Transcript transcript;
        if (row == "diarize") transcript = new Transcript { Text = "", Duration = TimeSpan.FromSeconds(durationSeconds) };
        else
        {
            phase = "recognition";
            var deferred = new DeferredTiming();
            GraphTiming.BeginPhase();
            transcript = new VadTranscriptionPipeline(recognizer!, recognitionVad!) { RecognizedChunk = deferred.Add }.Transcribe(file, "en", default);
            GraphTiming.EndPhase(1);
            phase = "alignment";
            using var replay = AudioFileSource.Open(file, default);
            GraphTiming.BeginPhase();
            transcript = deferred.Apply(VadChunker.Blocks(replay, default), replayVad!, transcript, aligner!, default);
            GraphTiming.EndPhase(2);
        }
        phase = "diarization";
        using var source = AudioFileSource.Open(file, default);
        GraphTiming.BeginPhase();
        var turns = diarizer!.DiarizeBlocks(VadChunker.Blocks(source, default), default);
        var merged = TranscriptDiarization.Merge(transcript, turns.Select(t => new ActiveTurn { Turn = t }), default);
        var document = merged with { Models = [.. merged.Models.Concat(diarizer.Models).DistinctBy(m => (m.Id, m.Revision))] };
        GraphTiming.EndPhase(3);
        phase = "complete";
        return document;
    }

    public void Dispose()
    {
        for (int i = owners.Count - 1; i >= 0; i--) owners[i].Dispose();
        owners.Clear();
    }

    private sealed class BorrowedTiming(Aligner primary, Func<OmniAsrAligner> fallback) : Aligner
    {
        public override ModelInfo Model => primary.Model;
        public override Capabilities Capabilities { get; } = new() { ForcedAlignment = true, Languages = ["en"] };
        public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language, CancellationToken cancellationToken) =>
            VadTranscriptionPipeline.ApplyTiming(audio, transcript, language, Candidates(), cancellationToken);
        private IEnumerable<Aligner> Candidates()
        {
            yield return primary;
            // Match the file helper's scoped fallback. Its reload cost is inside
            // end-to-end, never subtracted or hidden in graph time.
            using var temporary = fallback();
            yield return temporary;
        }
        public override Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken)
        {
            var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(w => new Word { Text = w, Language = language }).ToImmutableArray();
            return Align(audio, new Transcript { Text = text, Words = words, Language = language }, language, cancellationToken);
        }
    }
}
