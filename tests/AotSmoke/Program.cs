using OneSpeech;

var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native, Speaker = SpeakerLabel.Unassigned };
var original = new Transcript { Text = "test", Duration = TimeSpan.FromSeconds(1), Words = [word] };
var copy = Transcript.FromJson(original.ToJson());
if (copy.Text != original.Text || copy.Words[0] != word) throw new InvalidOperationException("Transcript round-trip failed.");
Console.WriteLine("Transcript JSON NativeAOT round-trip passed.");
var features = new OneSpeech.Audio.WhisperLogMelSpectrogram(128).Extract(new AudioBuffer(new float[400], 16000), padToWindow: false);
if (features.Length != 256 || features.Any(value => value != -1.5f)) throw new InvalidOperationException("Whisper feature smoke failed.");
Console.WriteLine("Whisper features NativeAOT smoke passed.");
