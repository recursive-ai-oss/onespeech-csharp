using System.Diagnostics;
using System.Globalization;
using OneAudio;
using OneSpeech;

if (args.Length != 4) { Console.Error.WriteLine("Usage: cli-smoke <installed-tool-path> <public-clip-one> <public-clip-two> <output-directory>"); return 2; }
string tool = Path.GetFullPath(args[0]), directory = Path.GetFullPath(args[3]);
Directory.CreateDirectory(directory);
var first = Audio.Load(args[1], new LoadOptions { SampleRate = 16000, Channels = 1 });
var second = Audio.Load(args[2], new LoadOptions { SampleRate = 16000, Channels = 1 });
float[] joined = new float[first.Samples.Length + 16000 + second.Samples.Length];
first.Samples.Span.CopyTo(joined); second.Samples.Span.CopyTo(joined.AsSpan(first.Samples.Length + 16000));
string arranged = Path.Combine(directory, "two-speaker.wav");
Write(arranged, joined, 1);
Console.WriteLine("Prepared synthetic two-speaker arrangement from public evaluation clips; no voice is synthesized.");
foreach (string input in new[] { args[1], arranged })
{
    string json = await Run(0, "transcribe", input, "--model", "openai/whisper-tiny", "--language", "en", "--format", "json");
    var transcript = Transcript.FromJson(json);
    if (string.IsNullOrWhiteSpace(transcript.Text)) throw new InvalidOperationException("Missing recognized speech.");
    string text = Path.Combine(directory, "written.txt");
    await File.WriteAllTextAsync(text, transcript.Text);
    var aligned = Transcript.FromJson(await Run(0, "align", input, text, "--language", "en", "--format", "json"));
    if (aligned.Words.Length == 0) throw new InvalidOperationException("Alignment has no words.");
    var diarized = Transcript.FromJson(await Run(0, "diarize", input, "--format", "json"));
    if (diarized.Turns.Length == 0) throw new InvalidOperationException("Diarization has no speech.");
    int speakers = diarized.Turns.Select(t => t.Speaker.Id).Distinct(StringComparer.Ordinal).Count();
    if (input == arranged && speakers < 2) throw new InvalidOperationException("Two-speaker arrangement produced fewer than two labels.");
    Console.WriteLine($"{Path.GetFileName(input)}: words={transcript.Words.Length}, aligned={aligned.Words.Length}, turns={diarized.Turns.Length}, speakers={speakers}");
    await Run(0, "enroll", "first", input);
    string identity = await Run(0, "identify", input, "first", input, "--threshold", "-1");
    if (!identity.StartsWith("first\t", StringComparison.Ordinal)) throw new InvalidOperationException("Self identification failed.");
    double score = double.Parse(await Run(0, "verify", input, input), CultureInfo.InvariantCulture);
    if (score < 0.9999) throw new InvalidOperationException("Self verification failed.");
    {
        string language = await Run(0, "detect-language", input, "--model", "openai/whisper-tiny");
        if (string.IsNullOrWhiteSpace(language)) throw new InvalidOperationException("No language result.");
    }
}
foreach (string format in new[] { "text", "srt", "vtt" })
{
    string path = Path.Combine(directory, "transcript." + format);
    await Run(0, "transcribe", args[1], "--model", "openai/whisper-tiny", "--language", "en", "--format", format, "--output", path);
    string result = await File.ReadAllTextAsync(path);
    if (string.IsNullOrWhiteSpace(result) || (format == "vtt" && !result.StartsWith("WEBVTT", StringComparison.Ordinal))) throw new InvalidOperationException("Missing output: " + format);
}
using (var aac = OneSpeech.Media.Tests.ContainerFixture.Aac(args[1]))
using (var matroska = OneSpeech.Media.Tests.ContainerFixture.Matroska(aac, CancellationToken.None))
{
    string path = Path.Combine(directory, "speech.mkv");
    await File.WriteAllBytesAsync(path, matroska.ToArray());
    var mediaTranscript = Transcript.FromJson(await Run(0, "transcribe", path, "--model", "openai/whisper-tiny", "--language", "en", "--format", "json"));
    if (string.IsNullOrWhiteSpace(mediaTranscript.Text)) throw new InvalidOperationException("Matroska fallback returned no speech.");
    Console.WriteLine("PASS Matroska media input with a video track before the audio track.");
}
await Run(0, "models", "list");
await Run(0, "models", "pull", "openai/whisper-tiny");
await Run(0, "--help");
await Run(0, "--version");
await Run(2, "transcribe", "--format", "not-a-format");
string gate = await Run(1, "models", "pull", "pyannote/segmentation-3.0");
if (!gate.Contains("ModelTokenRequiredException", StringComparison.Ordinal)) throw new InvalidOperationException("Missing typed gate error.");
Console.WriteLine("All CLI verbs, formats, installed tool, usage and gate refusals passed.");
return 0;

void Write(string path, float[] samples, int channels)
{
    using var file = File.Create(path);
    using var writer = new WavEncoder(file, new AudioFormat(16000, channels));
    writer.Write(samples); writer.Finish();
}

async Task<string> Run(int expected, params string[] command)
{
    var start = new ProcessStartInfo(tool) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
    // Gate acceptance is not part of this test; never read a user's credentials.
    start.Environment.Remove("ONESPEECH_MODEL_TOKEN");
    foreach (string value in command) start.ArgumentList.Add(value);
    using var process = Process.Start(start) ?? throw new InvalidOperationException("Cannot start installed tool.");
    Task<string> stdout = process.StandardOutput.ReadToEndAsync(), stderr = process.StandardError.ReadToEndAsync();
    await process.WaitForExitAsync();
    string output = await stdout, error = await stderr;
    if (process.ExitCode != expected) throw new InvalidOperationException($"{command[0]}: exit {process.ExitCode}, expected {expected}: {error}");
    if (expected == 0 && error.Length != 0) throw new InvalidOperationException("Unexpected stderr: " + error);
    Console.WriteLine($"PASS {command[0]} (exit {process.ExitCode})");
    return expected == 0 ? output.Trim() : error;
}
