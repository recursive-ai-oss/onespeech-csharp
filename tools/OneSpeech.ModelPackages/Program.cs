using System.Security.Cryptography;
using OneSpeech;
using OneSpeech.Catalog;

if (args.Length < 2 || args.Length > 3 || (args[0] == "--verify" && args.Length != 3)) { Console.Error.WriteLine("Usage: model-packages <model-id> <output-directory> [cache-directory] | --verify <model-id> <directory>"); return 2; }
bool verify = args[0] == "--verify";
string id = args[verify ? 1 : 0];
string output = Path.GetFullPath(args[verify ? 2 : 1]);
var entry = ModelCatalog.Default.Get(id);
if (entry.Tier != ModelTier.Tier1 || entry.Gated || id is not ("iic/speech_campplus_sv_en_voxceleb_16k" or "moonshine-ai/moonshine-tiny" or "moonshine-ai/moonshine-base" or "moonshine-ai/moonshine-streaming-tiny"))
    throw new InvalidOperationException("Only explicitly approved ungated tier-1 packages can be prepared.");
Directory.CreateDirectory(output);
var source = new HttpModelSource();
foreach (var file in entry.Files)
{
    string destination = Path.Combine(output, file.Sha256);
    if (!File.Exists(destination) && !verify)
    {
        string? cached = null;
        if (args.Length > 2)
        {
            string root = Path.GetFullPath(args[2]);
            // Common publisher cache layouts; each candidate is hash checked before use.
            string[] candidates = [Path.Combine(root, id.Split('/')[^1], file.Name), Path.Combine(root, file.Name)];
            cached = candidates.FirstOrDefault(File.Exists);
        }
        string temporary = destination + ".part";
        try
        {
            await using Stream input = cached is null ? await source.OpenReadAsync(entry, file, CancellationToken.None) : File.OpenRead(cached);
            await using (var target = File.Create(temporary))
            {
                var buffer = new byte[65536];
                long total = 0;
                int count;
                while ((count = await input.ReadAsync(buffer)) != 0)
                {
                    total += count;
                    if (total > file.Size) throw new InvalidDataException($"File exceeds catalog size: {file.Name}");
                    await target.WriteAsync(buffer.AsMemory(0, count));
                }
            }
            await CheckAsync(temporary, file);
            File.Move(temporary, destination);
        }
        finally { File.Delete(temporary); }
    }
    await CheckAsync(destination, file);
    Console.WriteLine($"{id}\t{file.Name}\t{file.Size}\t{file.Sha256}");
}
var allowed = entry.Files.Select(file => file.Sha256).ToHashSet(StringComparer.Ordinal);
if (Directory.EnumerateFileSystemEntries(output).Any(path => !File.Exists(path) || !allowed.Contains(Path.GetFileName(path))))
    throw new InvalidDataException("Prepared model directory contains unexpected content; use a directory containing only this model's catalog-pinned files.");
return 0;

static async Task CheckAsync(string path, CatalogFile file)
{
    if (new FileInfo(path).Length != file.Size) throw new InvalidDataException($"Incorrect size: {file.Name}");
    await using var stream = File.OpenRead(path);
    string sha = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream));
    if (sha != file.Sha256) throw new InvalidDataException($"Incorrect SHA-256: {file.Name}");
}
