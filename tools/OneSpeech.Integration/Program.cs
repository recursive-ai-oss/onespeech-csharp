using System.Diagnostics;
using System.Globalization;
using OneAudio;
using OneAudio.Aac;
using OneAudio.Mp4;
using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.Evaluation;

if (args is ["comparison-summary", var summaryRoot]) { Summary.Write(summaryRoot); return 0; }
if (args is ["comparison-clips", var clipInput, var clipOutput]) return FileComparison.PrepareClips(clipInput, clipOutput);
if (args is ["comparison-earnings", var earningsRoot, var earningsManifest]) return FileComparison.PrepareEarnings(earningsRoot, earningsManifest);
if (args is ["comparison-manifest", var corpus, var manifest]) return FileComparison.Prepare(corpus, manifest);
if (args.Length == 6 && args[0] is "comparison" or "comparison-greedy" or "comparison-speed") return FileComparison.Run(args);

using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
string? sourceRoot = Environment.GetEnvironmentVariable("ONESPEECH_INTEGRATION_SOURCE");
if (sourceRoot is not null) ModelStore.Default = new ModelStore(source: new DirectoryModelSource(sourceRoot));
if (args is ["prepare", var first, var second, var output])
{
    Directory.CreateDirectory(output);
    var a = Audio.Load(first, new LoadOptions { SampleRate = 16000, Channels = 1 });
    var b = Audio.Load(second, new LoadOptions { SampleRate = 16000, Channels = 1 });
    float[] samples = new float[a.Samples.Length + 16000 + b.Samples.Length];
    a.Samples.Span.CopyTo(samples); b.Samples.Span.CopyTo(samples.AsSpan(a.Samples.Length + 16000));
    using (var file = File.Create(Path.Combine(output, "voicemail.wav")))
    using (var wav = new WavEncoder(file, new AudioFormat(16000, 1))) { wav.Write(samples); wav.Finish(); }
    using (var file = File.Create(Path.Combine(output, "voicemail.m4a")))
    {
        var encoder = new AacEncoder(16000, 1);
        using var writer = Mp4AudioWriter.CreateAac(file, 16000, 1, encoder.AudioSpecificConfig, AacEncoder.PrimingFrames);
        int position = 0;
        while (position < samples.Length)
        {
            position += encoder.Write(samples.AsSpan(position));
            while (encoder.TryReadPacket(out var packet)) writer.AddSample(packet.Data.Span, 1024);
        }
        encoder.Finish();
        while (encoder.TryReadPacket(out var packet)) writer.AddSample(packet.Data.Span, 1024);
        writer.Finish(samples.Length);
    }
    string Truth(string path)
    {
        string id = Path.GetFileNameWithoutExtension(path);
        return File.ReadLines(Directory.GetFiles(Path.GetDirectoryName(path)!, "*.trans.txt").Single()).Single(l => l.StartsWith(id + " ", StringComparison.Ordinal))[(id.Length + 1)..];
    }
    File.WriteAllText(Path.Combine(output, "truth.txt"), Truth(first) + " " + Truth(second));
    File.WriteAllText(Path.Combine(output, "inputs.txt"), $"{Path.GetFileName(first)}\n{Path.GetFileName(second)}\nCC-BY-4.0, LibriSpeech, Panayotov et al.\nSpeaker boundary: {a.Samples.Length / 16000.0:R}–{a.Samples.Length / 16000.0 + 1:R} seconds\n");
    return 0;
}
if (args is ["pull", var id])
{
    Console.WriteLine((await ModelStore.Default.PullAsync(id, cancellationToken: cancellation.Token)).Directory);
    return 0;
}
if (args is [var operation, var path, var destination, var truthPath, var model] && operation is ("transcribe" or "transcribe-stream"))
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(destination))!);
    // Sampling is diagnostic only; it includes model loading and the immutable output document.
    using var process = Process.GetCurrentProcess();
    using var rss = new StreamWriter(destination + ".rss.tsv");
    object gate = new();
    var clock = Stopwatch.StartNew();
    await using var sampler = new Timer(_ => { lock (gate) { process.Refresh(); rss.WriteLine($"{clock.Elapsed.TotalSeconds.ToString("F1", CultureInfo.InvariantCulture)}\t{process.WorkingSet64}"); rss.Flush(); } }, null, TimeSpan.Zero, TimeSpan.FromSeconds(5));
    Transcript transcript;
    if (operation == "transcribe-stream")
    {
        await using var input = File.OpenRead(path);
        transcript = await Speech.TranscribeAsync(input, cancellation.Token, model: model == "default" ? null : model);
    }
    else transcript = Speech.Transcribe(path, cancellation.Token, model: model == "default" ? null : model);
    File.WriteAllText(destination + ".json", transcript.ToJson());
    File.WriteAllText(destination + ".txt", transcript.Text);
    var score = Metrics.Score(Normalizer.Normalize(File.ReadAllText(truthPath), "en"), Normalizer.Normalize(transcript.Text, "en"));
    Console.WriteLine($"text={transcript.Text}\nlanguage={transcript.Language}; words={transcript.Words.Length}; turns={transcript.Turns.Length}; speakers={transcript.Turns.Select(t => t.Speaker).Distinct().Count()}; timing={string.Join(',', transcript.Words.GroupBy(w => w.TimingMethod).Select(g => g.Key + ":" + g.Count()))}\nnormalizer=en/1; errors={score.Errors}; reference={score.Reference}; WER={score.Rate:P4}");
    return 0;
}
if (args is ["detect", var audioPath, var modelId])
{
    Console.WriteLine(Speech.DetectLanguage(audioPath, cancellation.Token, model: modelId));
    return 0;
}
Console.Error.WriteLine("Usage: prepare <first.flac> <second.flac> <output> | pull <id> | transcribe[-stream] <audio> <output-prefix> <truth.txt> <model-id|default> | detect <audio> <model-id>");
return 2;
