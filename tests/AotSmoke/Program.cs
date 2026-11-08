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
