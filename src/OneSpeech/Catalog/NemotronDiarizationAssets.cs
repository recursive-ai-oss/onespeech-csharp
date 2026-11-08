using System.Security.Cryptography;
using OneSpeech.Catalog.Conversion;

namespace OneSpeech.Catalog;

// These digests describe local-only artifacts. They are deliberately not catalog download/mirror URLs.
internal static class NemotronDiarizationAssets
{
    internal const string Id = "nvidia/Nemotron-3-Diarization";
    internal const string Revision = "f667ed73aee57d40cc39428eb768b4fd87a0a29e";
    internal const string WeightSha = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6";
    internal static readonly (string Name, long Size, string Sha)[] RuntimeFiles =
    [
        ("embedding.onemodel", 2098048, "f4f35ed4f8936119dea9d755906fe399afc01240caafe1a89de5aa76ef292f51"),
        ("diarization.onemodel", 395111072, "14c569267d6ab78b25c56e00864c62f0543ea083dea28bb6e76ea91baac0fd17"),
        ("silence.f32", 2048, "d4417b3c0eabdf7c47032fac2b5b5a7ee83d819a6ddda8fd8eaf74e2b5cc4ac7")
    ];
    internal static void VerifyPublisher(string directory, CancellationToken cancellationToken)
    {
        if (!Matches(Path.Combine(directory, "model.safetensors"), 396954592, WeightSha, cancellationToken))
            throw new ModelIntegrityException("Nemotron 3 Diarization requires the pinned publisher safetensors. Pull nvidia/Nemotron-3-Diarization again.");
    }
    internal static bool Matches(string path, long size, string sha, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!File.Exists(path)) return false;
        using var file = File.OpenRead(path);
        if (file.Length != size) return false;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] bytes = new byte[65536]; int read;
        while ((read = file.Read(bytes)) != 0) { cancellationToken.ThrowIfCancellationRequested(); hash.AppendData(bytes, 0, read); }
        return Convert.ToHexStringLower(hash.GetHashAndReset()) == sha;
    }
    internal static void VerifyRuntime(string directory, CancellationToken cancellationToken)
    {
        foreach (var file in RuntimeFiles)
            if (!Matches(Path.Combine(directory, file.Name), file.Size, file.Sha, cancellationToken))
                throw new ModelIntegrityException("Nemotron local runtime SHA-256 mismatch: " + file.Name + ". Pull the model again.");
    }
    // Called under ModelStore's per-model lease. Tier-2 outputs remain in this user's cache.
    internal static void EnsureRuntime(CatalogEntry entry, string directory, CancellationToken cancellationToken)
    {
        if (entry.Id != Id || entry.Revision != Revision || entry.Tier != ModelTier.Tier2 || entry.IsDefault ||
            entry.Files.Length != 1 || entry.Files[0].Sha256 != WeightSha)
            throw new ModelIntegrityException("Nemotron publisher pin or tier differs from the supported catalog entry.");
        if (RuntimeFiles.All(f => Matches(Path.Combine(directory, f.Name), f.Size, f.Sha, cancellationToken))) return;
        VerifyPublisher(directory, cancellationToken);
        string staging = Path.Combine(directory, ".convert-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(staging);
        try
        {
            NemotronDiarizationConverter.Convert(directory, staging, cancellationToken);
            VerifyRuntime(staging, cancellationToken);
            foreach (var file in RuntimeFiles) File.Move(Path.Combine(staging, file.Name), Path.Combine(directory, file.Name), overwrite: true);
        }
        finally { Directory.Delete(staging, recursive: true); }
    }
    internal const string Attribution = """
NVIDIA Nemotron 3 Diarization, NVIDIA Corporation.
Source: https://huggingface.co/nvidia/Nemotron-3-Diarization
Revision: f667ed73aee57d40cc39428eb768b4fd87a0a29e
Published parameters are unmodified. OneSpeech constructs local runtime graphs; these are not redistributed.

OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)

By exercising rights granted to you under this agreement, you accept and agree to its terms.

As used in this agreement, “Model Materials” means the materials provided to you under this agreement, consisting of: (1) one or more machine learning models (including architecture and parameters); and (2) all related artifacts (including associated data, documentation and software) that are provided to you hereunder.

Subject to your compliance with this agreement, permission is hereby granted, free of charge, to deal in the Model Materials without restriction, including under all copyright, patent, database, and trade secret rights included or embodied therein.

If you distribute any portion of the Model Materials, you shall retain in your distribution (1) a copy of this agreement, and (2) all copyright notices and other notices of origin included in the Model Materials that are applicable to your distribution.

If you file, maintain, or voluntarily participate in a lawsuit against any person or entity asserting that the Model Materials directly or indirectly infringe any patent or copyright, then all rights and grants made to you hereunder are terminated, unless that lawsuit was in response to a corresponding lawsuit first brought against you.

This agreement does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs generated by using the Model Materials.

THE MODEL MATERIALS ARE PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, NONINFRINGEMENT, ACCURACY, OR THE ABSENCE OF LATENT OR OTHER DEFECTS OR ERRORS, WHETHER OR NOT DISCOVERABLE, ALL TO THE GREATEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW.

YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF, INCLUDING WITHOUT LIMITATION ANY PERSON’S COPYRIGHTS OR OTHER RIGHTS INCLUDED OR EMBODIED IN THE MODEL MATERIALS; (2) OBTAINING ANY NECESSARY CONSENTS, PERMISSIONS OR OTHER RIGHTS REQUIRED FOR ANY USE OF THE MODEL MATERIALS; OR (3) PERFORMING ANY DUE DILIGENCE OR UNDERTAKING ANY OTHER INVESTIGATIONS INTO THE MODEL MATERIALS OR ANYTHING INCORPORATED OR EMBODIED THEREIN.

IN NO EVENT SHALL THE PROVIDERS OF THE MODEL MATERIALS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL MATERIALS, THE USE THEREOF OR OTHER DEALINGS THEREIN.
""";
}
