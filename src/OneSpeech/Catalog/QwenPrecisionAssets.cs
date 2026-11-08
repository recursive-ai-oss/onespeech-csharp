using System.Security.Cryptography;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Qwen3;

namespace OneSpeech.Catalog;

// Local precision conversion never downloads files or trusts persistent verification markers.
internal static class QwenPrecisionAssets
{
    internal static void EnsureBFloat16(CatalogEntry entry, string directory, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var lease = new FileStream(Path.Combine(directory, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        var files = entry.ConvertedFiles.Where(file => file.Name.EndsWith("-bf16.onemodel", StringComparison.Ordinal)).ToArray();
        bool missing = false;
        foreach (var file in files)
        {
            string path = Path.Combine(directory, file.Name);
            if (!File.Exists(path)) { missing = true; continue; }
            Verify(file, path, cancellationToken);
        }
        if (!missing) return;
        // Loading from a caller-supplied directory still requires the publisher pins before conversion.
        foreach (var file in entry.Files) Verify(file, Path.Combine(directory, file.Name), cancellationToken);
        string staging = Path.Combine(directory, ".convert-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(staging);
        try
        {
            QwenConverter.Convert(directory, staging, cancellationToken, Qwen3Precision.BFloat16);
            foreach (var file in files) Verify(file, Path.Combine(staging, file.Name), cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var file in files) File.Move(Path.Combine(staging, file.Name), Path.Combine(directory, file.Name), overwrite: true);
        }
        finally { Directory.Delete(staging, recursive: true); }
    }

    private static void Verify(CatalogFile file, string path, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var input = File.OpenRead(path);
        if (input.Length != file.Size || !Convert.ToHexStringLower(SHA256.HashData(input)).Equals(file.Sha256, StringComparison.OrdinalIgnoreCase))
            throw new ModelIntegrityException($"File '{file.Name}' differs from the catalog; remove it and retry the pull or conversion.");
        cancellationToken.ThrowIfCancellationRequested();
    }
}
