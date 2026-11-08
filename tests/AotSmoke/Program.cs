using OneSpeech;

var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native, Speaker = SpeakerLabel.Unassigned };
var original = new Transcript { Text = "test", Duration = TimeSpan.FromSeconds(1), Words = [word] };
var copy = Transcript.FromJson(original.ToJson());
if (copy.Text != original.Text || copy.Words[0] != word) throw new InvalidOperationException("Transcript round-trip failed.");

if (OneSpeech.Catalog.ModelCatalog.Default.Get("openai/whisper-tiny").Revision != "169d4a4341b33bc18d8881c4b69c2e104e1cc0af" ||
    OneSpeech.Catalog.ModelCatalog.Default.Get("iic/speech_campplus_sv_en_voxceleb_16k").ConvertedFiles[0].Sha256 != "f333a00d5d66b9bf3d7e7abf35f36ff4a01e6e8a0df235b89a125e9fbe3cc15c")
    throw new InvalidOperationException("Embedded catalog pins failed.");
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

SpeakerSmoke.Run();
Console.WriteLine("Speaker arithmetic and in-memory registry NativeAOT smoke passed.");

if (args is ["--camp" or "--camp-zh", var campGraph, var campAudio])
{
    var clip = OneAudio.Audio.Load(campAudio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    bool chinese = args[0] == "--camp-zh";
    var entry = OneSpeech.Catalog.ModelCatalog.Default.Get(chinese ? "iic/speech_campplus_sv_zh-cn_16k-common" : "iic/speech_campplus_sv_en_voxceleb_16k");
    var model = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, License = entry.License, Tier = entry.Tier, Attribution = entry.Attribution };
    using var camp = new OneSpeech.Embedders.CampPlusEmbedder(campGraph, model, dimension: chinese ? 192 : 512);
    float[] vector = new float[camp.Dimension];
    camp.Embed(new AudioBuffer(clip.Samples, 16000), vector, CancellationToken.None);
    Console.WriteLine("CAM++ real graph NativeAOT extraction passed.");
}

if (args is ["--ecapa", var ecapaGraph, var ecapaAudio])
{
    var entry = OneSpeech.Catalog.ModelCatalog.Default.Get("speechbrain/spkrec-ecapa-voxceleb");
    if (entry.ConvertedFiles[0].Sha256 != "f107ead463783cab70914fd0470cc6b42e2ec74b7a717bb42c1360879076c669")
        throw new InvalidOperationException("ECAPA graph pin failed.");
    var clip = OneAudio.Audio.Load(ecapaAudio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    var info = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, License = entry.License, Tier = entry.Tier, Attribution = entry.Attribution };
    using var model = new OneSpeech.Embedders.EcapaTdnnEmbedder(ecapaGraph, info);
    float[] vector = new float[192];
    model.Embed(new AudioBuffer(clip.Samples, 16000), vector, CancellationToken.None);
    Console.WriteLine("ECAPA real graph NativeAOT extraction passed.");
}

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
