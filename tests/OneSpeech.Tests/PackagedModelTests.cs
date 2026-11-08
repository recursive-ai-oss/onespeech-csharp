using System.Security.Cryptography;
using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class PackagedModelTests
{
    [Fact]
    public async Task PackagedPublisherBytesAreVerifiedBeforeUseAndNeverHideCorruption()
    {
        byte[] bytes = Guid.NewGuid().ToByteArray();
        string sha = Convert.ToHexStringLower(SHA256.HashData(bytes));
        string package = Path.Combine(AppContext.BaseDirectory, "onespeech-models", sha);
        string cache = Path.Combine(Path.GetTempPath(), "onespeech-package-" + Guid.NewGuid().ToString("N"));
        var entry = new CatalogEntry
        {
            Id = "publisher/package", Family = "test", Source = "https://huggingface.co/publisher/package",
            Revision = new string('a', 40), License = "MIT", Tier = ModelTier.Tier1, Attribution = "Synthetic fixture",
            Files = [new CatalogFile { Name = "weights", Size = bytes.Length, Sha256 = sha, UpstreamUrl = "https://huggingface.co/publisher/package/resolve/" + new string('a', 40) + "/weights" }]
        };
        var ct = TestContext.Current.CancellationToken;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(package)!);
            await File.WriteAllBytesAsync(package, bytes, ct);
            var store = new ModelStore(cache, new Unreachable(), new ModelCatalog([entry]));
            var resolved = await store.PullAsync(entry.Id, cancellationToken: ct);
            Assert.Equal(bytes, await File.ReadAllBytesAsync(Path.Combine(resolved.Directory, "weights"), ct));
            Directory.Delete(cache, true);
            bytes[0] ^= 1;
            await File.WriteAllBytesAsync(package, bytes, ct);
            await Assert.ThrowsAsync<ModelIntegrityException>(async () => await new ModelStore(cache, new Unreachable(), new ModelCatalog([entry])).PullAsync(entry.Id, cancellationToken: ct));
        }
        finally { File.Delete(package); if (Directory.Exists(cache)) Directory.Delete(cache, true); }
    }

    [Fact]
    public async Task Tier2NeverReadsPackageContent()
    {
        var entry = ModelCatalog.Default.Get("nvidia/parakeet-tdt-0.6b-v3");
        var source = new PackagedModelSource(new Unreachable());
        await Assert.ThrowsAsync<ModelNetworkException>(async () => await source.OpenReadAsync(entry, entry.Files[0], TestContext.Current.CancellationToken));
    }

    private sealed class Unreachable : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => throw new ModelNetworkException("Network disabled for this test.");
    }
}
