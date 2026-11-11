using System.Globalization;
using OneSpeech.Audio;
using OneSpeech.Diarization;

// Runs the native community pipeline on one recording and writes overlapping RTTM turns (same layout as the reference runner).
internal static class CommunityDiarizationProbe
{
    internal static void Run(string snapshot, string cache, string audio, string output, int threads)
    {
        if (File.Exists(output)) return;
        using var model = CommunityDiarizer.LoadLocal(snapshot, cache, threads);
        using var source = AudioFileSource.Open(audio, CancellationToken.None);
        var turns = model.DiarizeBlocks(VadChunker.Blocks(source, CancellationToken.None), CancellationToken.None);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        string id = Path.GetFileNameWithoutExtension(audio);
        using (var writer = new StreamWriter(output + ".partial"))
            foreach (var turn in turns)
                writer.WriteLine(string.Create(CultureInfo.InvariantCulture, $"SPEAKER {id} 1 {turn.Start.TotalSeconds:F9} {(turn.End - turn.Start).TotalSeconds:F9} <NA> <NA> {turn.Speaker.Id} <NA> <NA>"));
        File.Move(output + ".partial", output);
        Console.WriteLine(id + " complete");
    }
}
