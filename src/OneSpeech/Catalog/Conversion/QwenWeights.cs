using OneSpeech.Formats;

namespace OneSpeech.Catalog.Conversion;

// Safetensors shard indexes are data; no executable serialization formats are involved.
internal sealed class QwenWeights : IDisposable
{
    private readonly List<SafeTensorFile> files = [];
    private readonly Dictionary<string, SafeTensorFile> owners = new(StringComparer.Ordinal);
    internal IReadOnlyDictionary<string, SafeTensorInfo> Tensors => tensors;
    private readonly Dictionary<string, SafeTensorInfo> tensors = new(StringComparer.Ordinal);
    internal QwenWeights(string directory)
    {
        string[] paths = File.Exists(Path.Combine(directory, "model.safetensors"))
            ? [Path.Combine(directory, "model.safetensors")]
            : [Path.Combine(directory, "model-00001-of-00002.safetensors"), Path.Combine(directory, "model-00002-of-00002.safetensors")];
        try
        {
            foreach (string path in paths)
            {
                var file = new SafeTensorFile(File.OpenRead(path)); files.Add(file);
                foreach (var pair in file.Tensors)
                {
                    if (!tensors.TryAdd(pair.Key, pair.Value)) throw new ModelIntegrityException("Duplicate tensor across Qwen shards.");
                    owners.Add(pair.Key, file);
                }
            }
        }
        catch { Dispose(); throw; }
    }
    internal void CopyTensorTo(string name, Stream destination, CancellationToken cancellationToken) => owners[name].CopyTensorTo(name, destination, cancellationToken);
    internal void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken) => owners[name].CopyFloat32To(name, destination, cancellationToken);
    public void Dispose() { foreach (var file in files) file.Dispose(); }
}
