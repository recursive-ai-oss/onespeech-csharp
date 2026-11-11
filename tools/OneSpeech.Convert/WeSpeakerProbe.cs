using System.Buffers.Binary;
using System.Text.Json;
using OneSpeech.Diarization;

// Compares the native WeSpeaker embedding with outputs recorded from the released reference on fixed clips.
internal static class WeSpeakerProbe
{
    private static float[] ReadF32(string path)
    {
        byte[] b = File.ReadAllBytes(path); var r = new float[b.Length / 4];
        for (int i = 0; i < r.Length; i++) r[i] = BinaryPrimitives.ReadSingleLittleEndian(b.AsSpan(i * 4, 4));
        return r;
    }

    internal static void Run(string checkpoint, string cache, string probes)
    {
        Directory.CreateDirectory(cache);
        string graphPath = Path.Combine(cache, "wespeaker-frames.onemodel");
        WeSpeakerResNetConverter converter;
        using (var input = File.OpenRead(checkpoint)) converter = new WeSpeakerResNetConverter(TensorCheckpoint.Read(input, segmentation: true));
        converter.Save(graphPath);
        var (w, b) = converter.Projection();
        using var embedder = new WeSpeakerEmbedder(graphPath, w, b, 4, default);
        using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(probes, "manifest.json")));
        foreach (var c in manifest.RootElement.GetProperty("cases").EnumerateArray())
        {
            if (!c.TryGetProperty("output", out var outName)) { Console.WriteLine(c.GetProperty("label").GetString() + " error " + c.GetProperty("error").GetString()); continue; }
            float[] audio = ReadF32(Path.Combine(probes, c.GetProperty("input").GetString()!));
            float[] expected = ReadF32(Path.Combine(probes, outName.GetString()!));
            float[] frames = embedder.Frames(audio, out int steps, default);
            float[] weights = [];
            if (c.TryGetProperty("weights_len", out var wl))
            {
                int n = wl.GetInt32(); weights = new float[n];
                string label = c.GetProperty("label").GetString()!;
                for (int i = 0; i < n; i++) weights[i] = label.Contains("-half-", StringComparison.Ordinal) ? (i < n / 2 ? 1 : 0) : 1;
            }
            float[] ours = embedder.Project(WeSpeakerEmbedder.Pool(frames, steps, weights));
            double dot = 0, na = 0, nb = 0, max = 0;
            for (int i = 0; i < ours.Length; i++) { dot += ours[i] * (double)expected[i]; na += ours[i] * (double)ours[i]; nb += expected[i] * (double)expected[i]; max = Math.Max(max, Math.Abs(ours[i] - expected[i])); }
            Console.WriteLine($"{c.GetProperty("label").GetString()} steps={steps} cos={dot / Math.Sqrt(na * nb):F9} maxabs={max:E3} refnorm={Math.Sqrt(nb):F3}");
        }
    }
}
