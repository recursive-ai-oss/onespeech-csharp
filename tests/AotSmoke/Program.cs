using OneSpeech;

var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native, Speaker = SpeakerLabel.Unassigned };
var original = new Transcript { Text = "test", Duration = TimeSpan.FromSeconds(1), Words = [word] };
var copy = Transcript.FromJson(original.ToJson());
if (copy.Text != original.Text || copy.Words[0] != word) throw new InvalidOperationException("Transcript round-trip failed.");
Console.WriteLine("Transcript JSON NativeAOT round-trip passed.");
var features = new OneSpeech.Audio.WhisperLogMelSpectrogram(128).Extract(new AudioBuffer(new float[400], 16000), padToWindow: false);
if (features.Length != 256 || features.Any(value => value != -1.5f)) throw new InvalidOperationException("Whisper feature smoke failed.");
Console.WriteLine("Whisper features NativeAOT smoke passed.");
string? tinyDirectory = Environment.GetEnvironmentVariable("ONESPEECH_WHISPER_TINY");
string? clipPath = Environment.GetEnvironmentVariable("ONESPEECH_SMOKE_CLIP");
if (tinyDirectory is not null && clipPath is not null)
{
    using var recognizer = OneSpeech.Whisper.WhisperRecognizer.Load(tinyDirectory);
    var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    var transcript = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
    if (transcript.Text != " Stuffed into you, his belly, cowtled him." || transcript.Words.Length == 0) throw new InvalidOperationException("Whisper tiny NativeAOT transcription failed.");
    Console.WriteLine("Whisper tiny NativeAOT transcription passed.");
    using var beam = OneSpeech.Whisper.WhisperRecognizer.Load(tinyDirectory, new OneSpeech.Whisper.WhisperOptions { BeamSize = 1, Temperatures = [0], WordTimestamps = false });
    if (beam.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None).Text != transcript.Text) throw new InvalidOperationException("Whisper beam-width-one smoke failed.");
    Console.WriteLine("Whisper beam NativeAOT smoke passed.");
}
