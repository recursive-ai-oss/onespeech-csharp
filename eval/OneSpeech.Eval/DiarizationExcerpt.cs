using System.Collections.Immutable;
using System.Text.Json;
using OneSpeech.Audio;
using OneSpeech.Diarization;
using OneSpeech.Pipeline;
using OneSpeech.Vad;

namespace OneSpeech.Evaluation;

internal static class DiarizationExcerpt
{
    internal static void Run(string graph, string root, string id, string output, DiarizationRun.Profile profile)
    {
        var entry = DiarizationRun.ReadEntry(root, id);
        if (entry.Split != "test" || id is not ("ES2004a" or "ES2004b")) throw new ArgumentException("Excerpt smoke requires one of the fixed AMI test meetings.");
        DiarizationRun.VerifyGraph(graph); DiarizationRun.VerifyEntry(root, entry);
        string crop = Path.Combine(root, "prepared", id + "-first300s.wav"); Crop(Path.Combine(root, entry.Audio), crop);
        using var embedder = DiarizationRun.LoadEmbedder(graph); using var vad = new SileroVad();
        var diarizer = new ClusteringDiarizer(embedder, vad, profile.Options);
        var pipeline = new VadTranscriptionPipeline(new TimingFixture(), vad, null, null, diarizer);
        var transcript = pipeline.Transcribe(crop, "en", CancellationToken.None);
        if (transcript.Duration > TimeSpan.FromMinutes(5) || transcript.Words.IsEmpty || !transcript.Words.Any(w => w.Speaker is not null && w.Speaker.Id != SpeakerLabel.Unassigned.Id)) throw new InvalidOperationException("Excerpt did not produce assigned timed fixture words.");
        Directory.CreateDirectory(output); string target = Path.Combine(output, id + "-excerpt.transcript.json"); File.WriteAllText(target, transcript.ToJson());
        var roundtrip = Transcript.FromJson(File.ReadAllText(target));
        if (roundtrip.Words.Length != transcript.Words.Length) throw new InvalidOperationException("Excerpt JSON lost words.");
        long maximumTickError = 0; string? firstTimeDifference = null;
        for (int i = 0; i < transcript.Words.Length; i++)
        {
            var original = transcript.Words[i]; var copy = roundtrip.Words[i];
            if (original.Start is not {} expectedStart || original.End is not {} expectedEnd || copy.Start is not {} actualStart || copy.End is not {} actualEnd ||
                copy with { Start = original.Start, End = original.End } != original)
                throw new InvalidOperationException("Excerpt JSON changed non-timing word fields or lost timings.");
            long error = Math.Max(Math.Abs(actualStart.Ticks - expectedStart.Ticks), Math.Abs(actualEnd.Ticks - expectedEnd.Ticks));
            if (error != 0) firstTimeDifference ??= $"word={i}, originalStartTicks={expectedStart.Ticks}, decodedStartTicks={actualStart.Ticks}, originalEndTicks={expectedEnd.Ticks}, decodedEndTicks={actualEnd.Ticks}";
            maximumTickError = Math.Max(maximumTickError, error);
        }
        // Version-1 double-second JSON may lose one 100 ns tick. Report it; do not alter timings to hide it.
        if (maximumTickError > 1) throw new InvalidOperationException("Excerpt JSON timing drift exceeds one tick.");
        Console.WriteLine($"JSON maximum timing drift: {maximumTickError} ticks; first difference: {firstTimeDifference}");
        using var file = File.Create(Path.Combine(output, id + "-excerpt.summary.json")); using var w = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        w.WriteStartObject(); w.WriteString("meeting", id); w.WriteString("profile", profile.Name); w.WriteString("sourceAudioSha256", entry.AudioHash); w.WriteString("excerptAudioSha256", DiarizationRun.HashFile(crop));
        w.WriteString("graphSha256", DiarizationRun.GraphHash); w.WriteString("transcript", target); w.WriteNumber("durationSeconds", transcript.Duration.TotalSeconds); w.WriteNumber("words", transcript.Words.Length); w.WriteNumber("jsonMaximumTimingErrorTicks", maximumTickError); w.WriteString("jsonFirstTimeDifference", firstTimeDifference);
        w.WriteNumber("assignedWords", transcript.Words.Count(x => x.Speaker is not null && x.Speaker.Id != SpeakerLabel.Unassigned.Id)); w.WriteNumber("unassignedWords", transcript.Words.Count(x => x.Speaker == SpeakerLabel.Unassigned));
        w.WriteNumber("nullTimeWords", transcript.Words.Count(x => x.Start is null || x.End is null)); w.WriteNumber("turns", transcript.Turns.Length); w.WriteNumber("speakers", transcript.Turns.Select(t => t.Speaker.Id).Distinct().Count());
        w.WriteString("recognizer", "word-timing-fixture; generated clock probes, NOT recognized text, WER, or word-alignment quality"); w.WriteEndObject();
        Console.WriteLine($"Excerpt: {transcript.Words.Length} fixture words, {transcript.Words.Count(x => x.Speaker is not null && x.Speaker.Id != SpeakerLabel.Unassigned.Id)} assigned; {transcript.Duration.TotalSeconds} seconds.");
    }
    private static void Crop(string path, string output)
    {
        using var source = AudioFileSource.Open(path, CancellationToken.None); if (source.Channels != 1) throw new InvalidDataException("Prepared excerpt source must be mono.");
        using var file = File.Create(output); using var writer = new BinaryWriter(file);
        writer.Write("RIFF"u8); writer.Write(0); writer.Write("WAVEfmt "u8); writer.Write(16); writer.Write((short)1); writer.Write((short)1); writer.Write(16000); writer.Write(32000); writer.Write((short)2); writer.Write((short)16); writer.Write("data"u8); writer.Write(0);
        var buffer = new float[16000]; int frames = 0;
        while (frames < 300 * 16000)
        {
            int count = source.Read(buffer.AsSpan(0, Math.Min(buffer.Length, 300 * 16000 - frames)), CancellationToken.None); if (count == 0) break;
            for (int i = 0; i < count; i++) writer.Write((short)Math.Clamp(Math.Round(buffer[i] * 32768.0), short.MinValue, short.MaxValue)); frames += count;
        }
        file.Position = 4; writer.Write(36 + frames * 2); file.Position = 40; writer.Write(frames * 2);
    }
    private sealed class TimingFixture : SpeechRecognizer
    {
        public override ModelInfo Model { get; } = new() { Id = "word-timing-fixture", Source = "https://recursiveai.org/onespeech/", Revision = "1", License = "MIT", Tier = ModelTier.Tier1, Attribution = "Generated clock probes, not recognized words or an ASR model." };
        public override Capabilities Capabilities { get; } = new() { Languages = ["en"], NativeTimestamps = true };
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            var words = ImmutableArray.CreateBuilder<Word>();
            for (double second = 0; second + 0.2 < audio.Duration.TotalSeconds; second++)
            { cancellationToken.ThrowIfCancellationRequested(); words.Add(new Word { Text = "fixture ", Start = TimeSpan.FromSeconds(second + 0.1), End = TimeSpan.FromSeconds(second + 0.2), TimingMethod = TimingMethod.Native }); }
            string text = string.Concat(words.Select(w => w.Text)); var list = words.ToImmutable();
            return new Transcript { Text = text, Duration = audio.Duration, Language = "en", Words = list, Segments = [new Segment { Text = text, Words = list, Start = TimeSpan.Zero, End = audio.Duration, Language = "en" }], Models = [Model] };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => Transcribe(audio, language, cancellationToken).Segments;
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) { cancellationToken.ThrowIfCancellationRequested(); return "en"; }
    }
}
