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

SpeakerSmoke.Run();
Console.WriteLine("Speaker arithmetic and in-memory registry NativeAOT smoke passed.");

if (args is ["--camp", var campGraph, var campAudio])
{
    var clip = OneAudio.Audio.Load(campAudio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    var model = new ModelInfo { Id = "iic/speech_campplus_sv_en_voxceleb_16k", Source = "https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k", Revision = "032b8131a7ad812f87061955ca974c99060c5a03", License = "Apache-2.0", Tier = ModelTier.Tier1, Attribution = "CAM++ published checkpoint." };
    using var camp = new OneSpeech.Embedders.CampPlusEmbedder(campGraph, model);
    float[] vector = new float[512];
    camp.Embed(new AudioBuffer(clip.Samples, 16000), vector, CancellationToken.None);
    Console.WriteLine("CAM++ real graph NativeAOT extraction passed.");
}
