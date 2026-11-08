using System.Security.Cryptography;
using OneSpeech.Catalog;
using OneSpeech.Moonshine;
using OneSpeech.Whisper;

internal static class Int8Conversion
{
    internal static void Run(string input, string output)
    {
        string[] provenance = File.ReadAllLines(Path.Combine(input, "upstream.txt"));
        if (provenance.Length != 3 || provenance[0].EndsWith("-int8", StringComparison.Ordinal)) throw new InvalidDataException("Expected original publisher checkpoint provenance.");
        var entry = ModelCatalog.Default.Get(provenance[0] + "-int8");
        if (entry.Family == "whisper") _ = WhisperCheckpoint.Provenance(provenance);
        else _ = MoonshineProvenance.Verify(input, converted: false, CancellationToken.None);
        var upstream = entry.Files.Single(f => f.Name == "model.safetensors");
        using (var stream = File.OpenRead(Path.Combine(input, upstream.Name)))
            if (stream.Length != upstream.Size || Convert.ToHexStringLower(SHA256.HashData(stream)) != upstream.Sha256)
                throw new InvalidDataException("Publisher weight digest differs from the catalog.");
        Directory.CreateDirectory(output);
        foreach (string file in Directory.EnumerateFiles(input).Where(f => f.EndsWith(".json", StringComparison.Ordinal) || Path.GetFileName(f) == "merges.txt"))
            File.Copy(file, Path.Combine(output, Path.GetFileName(file)), overwrite: true);
        RuntimeConversion.Convert(entry, input, output, CancellationToken.None);
        RuntimeConversion.WriteProvenance(entry, output);
        foreach (var file in entry.ConvertedFiles)
        {
            using var stream = File.OpenRead(Path.Combine(output, file.Name));
            string digest = Convert.ToHexStringLower(SHA256.HashData(stream));
            if (stream.Length != file.Size || digest != file.Sha256) throw new InvalidDataException("Converted file differs from the catalog: " + file.Name);
            Console.WriteLine($"{file.Name} {stream.Length} {digest}");
        }
    }
}
