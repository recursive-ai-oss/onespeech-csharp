using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Media;
using OneSpeech.Media.Tests;
using RecursiveAI.OneMedia.Captions;

MediaInput.Register();
var words = new Word[] { new() { Text = "Caption", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Speaker = new() { Id = "speaker" } } };
var transcript = new Transcript { Text = "Caption", Words = [.. words] };
foreach (var text in new[] { transcript.ToSrt(), transcript.ToVtt(), transcript.ToAss(), transcript.ToTtml() })
    if (SubtitleFormats.Read(System.Text.Encoding.UTF8.GetBytes(text)).Cues.Count != 1) return 1;
if (transcript.ToSubtitles().ToTranscript().Words.Length != 1) return 2;
using var aac = ContainerFixture.Aac();
using var mkv = ContainerFixture.Matroska(aac, CancellationToken.None);
using (var source = AudioFileSource.Open(mkv, true, CancellationToken.None))
    if (source.Read(new float[4096], CancellationToken.None) == 0) return 3;
string root = Path.Combine(Path.GetTempPath(), "caption-smoke-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(root);
try
{
    string input = Path.Combine(root, "in.mp4"), output = Path.Combine(root, "out.mp4");
    File.WriteAllBytes(input, aac.ToArray());
    transcript.MuxCaptions(input, output);
    using var media = RecursiveAI.OneMedia.Media.Open(output);
    if (media.ContainerSubtitles.Count != 1 || media.ContainerSubtitles[0].Document.Cues[0].Text != "Caption") return 4;
}
finally { Directory.Delete(root, true); }
using var invalid = new MemoryStream("invalid container"u8.ToArray());
try
{
    await foreach (var _ in MediaInput.TranscribeStream(invalid)) { }
    return 5;
}
catch (RecursiveAI.OneMedia.MediaException) { }
Console.WriteLine("Media AOT: decode, cues, four writers, stream refusal and caption mux passed.");
return 0;
