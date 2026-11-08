using OneSpeech;

var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native, Speaker = SpeakerLabel.Unassigned };
var original = new Transcript { Text = "test", Duration = TimeSpan.FromSeconds(1), Words = [word] };
var copy = Transcript.FromJson(original.ToJson());
if (copy.Text != original.Text || copy.Words[0] != word) throw new InvalidOperationException("Transcript round-trip failed.");
Console.WriteLine("Transcript JSON NativeAOT round-trip passed.");
if (OneSpeech.Catalog.ModelCatalog.Default.Entries.Length != 6) throw new InvalidOperationException("Embedded catalog parsing failed.");
Console.WriteLine("Embedded catalog NativeAOT parsing passed.");
