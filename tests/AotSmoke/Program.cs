using OneSpeech;

var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native, Speaker = SpeakerLabel.Unassigned };
var original = new Transcript { Text = "test", Duration = TimeSpan.FromSeconds(1), Words = [word] };
var copy = Transcript.FromJson(original.ToJson());
if (copy.Text != original.Text || copy.Words[0] != word) throw new InvalidOperationException("Transcript round-trip failed.");

if (OneSpeech.Catalog.ModelCatalog.Default.Entries.Length != 6) throw new InvalidOperationException("Embedded catalog parsing failed.");
Console.WriteLine("Embedded catalog NativeAOT parsing passed.");
using var vad = new OneSpeech.Vad.SileroVad();
if (!vad.Detect(new AudioBuffer(new float[16000], 16000), CancellationToken.None).IsEmpty)
    throw new InvalidOperationException("Silero VAD reported speech on silence.");
Console.WriteLine("Transcript JSON and embedded Silero VAD NativeAOT smoke passed.");

var diarized = OneSpeech.Diarization.TranscriptDiarization.Merge(copy,
    [new OneSpeech.Diarization.ActiveTurn { Turn = new Turn { Speaker = new SpeakerLabel { Id = "speaker-1" }, Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Text = "" } }]);
if (diarized.Words[0].Speaker?.Id != "speaker-1" || OneSpeech.Diarization.ChannelDiarizer.IsSeparated(new AudioBuffer(new float[3200], 16000, 2)))
    throw new InvalidOperationException("Diarization NativeAOT smoke failed.");
Console.WriteLine("Diarization merge and channel detection NativeAOT smoke passed.");

using var smokeEmbedder = new SmokeEmbedder(vad.Model);
var clustering = new OneSpeech.Diarization.ClusteringDiarizer(smokeEmbedder, new SmokeVad(vad.Model),
    new OneSpeech.Diarization.ClusteringOptions { NumSpeakers = 1 });
if (clustering.Diarize(new AudioBuffer(new float[3200], 16000), CancellationToken.None).Length != 1)
    throw new InvalidOperationException("Clustering NativeAOT smoke failed.");
Console.WriteLine("Clustering NativeAOT smoke passed.");

internal sealed class SmokeEmbedder(ModelInfo model) : SpeakerEmbedder
{
    public override ModelInfo Model => model;
    public override int Dimension => 2;
    public override int SampleRate => 16000;
    public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
    { cancellationToken.ThrowIfCancellationRequested(); destination[0] = 1; destination[1] = 0; }
    public override void Dispose() { }
}
internal sealed class SmokeVad(ModelInfo model) : VoiceActivityDetector
{
    public override ModelInfo Model => model;
    public override System.Collections.Immutable.ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
    { cancellationToken.ThrowIfCancellationRequested(); return [new SpeechInterval { Start = TimeSpan.Zero, End = audio.Duration }]; }
}
