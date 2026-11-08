using System.Diagnostics;
using System.Globalization;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Embedders;
using OneSpeech.Evaluation;
using OneSpeech.Live;
using OneSpeech.Speakers;
using OneSpeech.Vad;

// WAV and RTTM are prepared evaluation data, never copied to the output directory.
if (args.Length != 6) throw new ArgumentException("graph.onemodel input.wav reference.rttm en|zh threshold output-prefix");
string id = Path.GetFileNameWithoutExtension(args[1]);
bool chinese = args[3] == "zh";
double threshold = double.Parse(args[4], CultureInfo.InvariantCulture);
var entry = ModelCatalog.Default.Get(chinese ? "iic/speech_campplus_sv_zh-cn_16k-common" : "iic/speech_campplus_sv_en_voxceleb_16k");
using var model = new CampPlusEmbedder(args[0], new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, Tier = ModelTier.Tier1, License = entry.License, Attribution = entry.Attribution }, chinese ? 192 : 512);
using var counted = new MeasuredEmbedder(model);
using var vad = new SileroVad();
var recognizer = new ActivityRecognizer(vad.Model, args[3]);
using var recognized = new UtteranceLiveSession(recognizer, vad, args[3]);
using var inner = new MeasuredSession(recognized);
using var session = new OnlineDiarizationSession(inner, counted, threshold);
using var source = AudioFileSource.Open(args[1], CancellationToken.None);
if (source.Channels != 1) throw new InvalidDataException("Prepared input must be mono.");
var buffer = new float[1600]; // 100 ms source chunks.
var hypotheses = new List<DiarizationSegment>();
long bytes = 0, chunks = 0, emptyBytes = 0, emptyChunks = 0, addedBytes = 0, addedEmptyBytes = 0;
double delay = 0, worst = 0;
int turnCount = 0;
void Events(IEnumerable<LiveEvent> events)
{
    foreach (var update in events)
        if (update is SpeakerTurnEvent turn)
        {
            if (turn.Turn.Speaker != SpeakerLabel.Unassigned)
                hypotheses.Add(new(id, "1", turn.Turn.Speaker.Id, turn.Turn.Start.TotalSeconds, turn.Turn.End.TotalSeconds));
            turnCount++;
            double latency = (turn.AudioTime - turn.Turn.End).TotalSeconds;
            delay += latency; worst = Math.Max(worst, latency);
        }
}
int length;
while ((length = source.Read(buffer, CancellationToken.None)) != 0)
{
    var audio = new AudioBuffer(buffer.AsMemory(0, length), 16000);
    long before = GC.GetAllocatedBytesForCurrentThread(), inherited = inner.Bytes, embedded = counted.Bytes;
    var updates = session.Push(audio);
    long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
    long added = allocated - (inner.Bytes - inherited) - (counted.Bytes - embedded);
    bytes += allocated; chunks++; addedBytes += added;
    if (chunks > 100 && updates.Length == 0) { emptyBytes += allocated; emptyChunks++; addedEmptyBytes += added; }
    Events(updates);
}
Events(session.Complete());
using var reference = File.OpenRead(args[2]);
var score = DiarizationMetrics.Score(Rttm.Read(reference), hypotheses);
using (var output = new StreamWriter(args[5] + ".rttm"))
    foreach (var segment in hypotheses)
        output.WriteLine(FormattableString.Invariant($"SPEAKER {id} 1 {segment.Start:F7} {segment.End - segment.Start:F7} <NA> <NA> {segment.Speaker} <NA> <NA>"));
// The centroid-only loop isolates clustering state from model inference and emitted snapshots.
var cluster = new OnlineSpeakerClusterer(model.Dimension, minimumSimilarity: threshold);
cluster.Assign(counted.Last);
for (int i = 0; i < 1000; i++) cluster.Assign(counted.Last);
long cb = GC.GetAllocatedBytesForCurrentThread();
for (int i = 0; i < 10000; i++) cluster.Assign(counted.Last);
long clusterBytes = GC.GetAllocatedBytesForCurrentThread() - cb;
string report = FormattableString.Invariant($"{id}\t{threshold}\t{score.Der * 100:F5}\t{score.Jer * 100:F5}\t{chunks}\t{hypotheses.Count}\t{delay / Math.Max(1, turnCount):F5}\t{worst:F5}\t{bytes}\t{counted.Bytes}\t{counted.Milliseconds:F3}\t{emptyBytes}\t{emptyChunks}\t{clusterBytes}\n");
File.WriteAllText(args[5] + ".tsv", "id\tthreshold\tDER\tJER\tchunks\tassigned_turns\tmean_turn_end_delay_s\tmax_turn_end_delay_s\tpush_bytes_including_model_events\tmodel_bytes\tmodel_ms_indicative\tempty_chunk_bytes\tempty_chunks\tcluster_10000_bytes\n" + report);
File.WriteAllText(args[5] + ".alloc.tsv", FormattableString.Invariant($"added_nonmodel_bytes_including_emitted_payloads\tadded_nonemitting_bytes\tchunks\n{addedBytes}\t{addedEmptyBytes}\t{chunks}\n"));
Console.Write(report);

sealed class ActivityRecognizer(ModelInfo info, string language) : SpeechRecognizer
{
    public override ModelInfo Model => info;
    public override Capabilities Capabilities => new() { Languages = [language] };
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => language;
    public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken) => new() { Text = "speech", Language = language, Words = [new Word { Text = "speech", Start = TimeSpan.Zero, End = audio.Duration }] };
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => [];
}
sealed class MeasuredEmbedder(SpeakerEmbedder model) : SpeakerEmbedder
{
    public long Bytes { get; private set; }
    public double Milliseconds { get; private set; }
    public float[] Last { get; } = new float[model.Dimension];
    public override int Dimension => model.Dimension;
    public override int SampleRate => model.SampleRate;
    public override ModelInfo Model => model.Model;
    public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
    {
        long before = GC.GetAllocatedBytesForCurrentThread(), time = Stopwatch.GetTimestamp();
        model.Embed(audio, destination, cancellationToken);
        Milliseconds += Stopwatch.GetElapsedTime(time).TotalMilliseconds;
        Bytes += GC.GetAllocatedBytesForCurrentThread() - before;
        destination.CopyTo(Last);
    }
    public override void Dispose() => Array.Clear(Last);
}
// Attribute allocations in the same pass, not by subtracting model warmup in a later replay.
sealed class MeasuredSession(LiveSession inner) : LiveSession
{
    public long Bytes { get; private set; }
    protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        var updates = inner.Push(audio, cancellationToken);
        Bytes += GC.GetAllocatedBytesForCurrentThread() - before;
        Forward(updates);
    }
    protected override void Finish(CancellationToken cancellationToken) => Forward(inner.Complete(cancellationToken));
    private void Forward(System.Collections.Immutable.ImmutableArray<LiveEvent> updates)
    {
        foreach (var update in updates)
            switch (update)
            {
                case PartialEvent partial: PublishPartial(partial.Segment); break;
                case FinalEvent final: PublishFinal(final.Segment); break;
                case EndpointEvent endpoint: PublishEndpoint(endpoint.End, endpoint.Reason); break;
                default: throw new InvalidOperationException("Expected recognition-only session.");
            }
    }
}
