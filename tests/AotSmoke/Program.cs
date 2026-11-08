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

SpeakerSmoke.Run();
Console.WriteLine("Speaker arithmetic and in-memory registry NativeAOT smoke passed.");
