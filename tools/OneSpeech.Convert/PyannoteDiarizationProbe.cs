using System.Globalization;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Embedders;

internal static class PyannoteDiarizationProbe
{
    internal static async Task Run(string snapshot, string cache, string embedder, string audio, string output)
    {
        if (File.Exists(output)) return;
        using var segmentation = PyannoteSegmentation.LoadLocal(snapshot, cache, threads: 4);
        using var embedding = await CampPlusEmbedder.LoadAsync(new ModelStore(cache), embedder, threads: 4);
        using var model = new PyannoteDiarizer(segmentation, embedding);
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
