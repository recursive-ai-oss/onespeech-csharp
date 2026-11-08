using System.Security.Cryptography;
using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class LazyRuntimeTests
{
    [Fact]
    public void LargeWhisperDecoderIsASinglePinnedFile()
    {
        var entry = ModelCatalog.Default.Get("openai/whisper-large-v3");
        var decoder = Assert.Single(entry.ConvertedFiles.Where(file => file.Name.StartsWith("decoder", StringComparison.Ordinal)));
        Assert.Equal("decoder.onemodel", decoder.Name);
        Assert.Equal(3206873536L, decoder.Size);
        Assert.Equal("73aecc737627012e80d8e59983b3fa9aba8e95621b4a54adecc3717377489571", decoder.Sha256);
    }

    [Fact]
    public async Task Fp32PullIgnoresMissingAndCorruptBFloat16Files()
    {
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        byte[] bytes = [1, 2, 3];
        var file = new CatalogFile { Name = "model.safetensors", Size = bytes.Length, Sha256 = System.Convert.ToHexStringLower(SHA256.HashData(bytes)), UpstreamUrl = "https://huggingface.co/publisher/model/resolve/" + new string('a', 40) + "/model.safetensors" };
        var fp32 = file with { Name = "encoder.onemodel", UpstreamUrl = "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/test/encoder.onemodel" };
        var bf16 = fp32 with { Name = "encoder-bf16.onemodel" };
        var entry = new CatalogEntry { Id = "publisher/model", Family = "qwen3-asr", Source = "https://huggingface.co/publisher/model", Revision = new string('a', 40), Tier = ModelTier.Tier1, License = "MIT", Attribution = "Synthetic fixture", Files = [file], ConvertedFiles = [fp32, bf16] };
        try
        {
            var initial = await new ModelStore(root, new BytesSource(bytes), new([entry with { ConvertedFiles = [] }])).PullAsync(entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            File.WriteAllBytes(Path.Combine(initial.Directory, fp32.Name), bytes);
            var store = new ModelStore(root, new RejectingSource(), new([entry]));
            var resolved = await store.PullAsync(entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(2, resolved.Info.Files.Length);
            Assert.False(File.Exists(Path.Combine(resolved.Directory, bf16.Name)));
            File.WriteAllBytes(Path.Combine(resolved.Directory, bf16.Name), [9]);
            await new ModelStore(root, new RejectingSource(), new([entry])).PullAsync(entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Throws<ModelIntegrityException>(() => QwenPrecisionAssets.EnsureBFloat16(entry, resolved.Directory, TestContext.Current.CancellationToken));
            Assert.Equal(bytes, File.ReadAllBytes(Path.Combine(resolved.Directory, fp32.Name)));
            File.Delete(Path.Combine(resolved.Directory, bf16.Name));
            File.WriteAllBytes(Path.Combine(resolved.Directory, file.Name), [9]);
            Assert.Throws<ModelIntegrityException>(() => QwenPrecisionAssets.EnsureBFloat16(entry, resolved.Directory, TestContext.Current.CancellationToken));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void BFloat16CacheHitVerifiesWithoutPublisherOrConversion()
    {
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            byte[] bytes = [1, 2, 3];
            var file = new CatalogFile { Name = "encoder-bf16.onemodel", Size = 3, Sha256 = System.Convert.ToHexStringLower(SHA256.HashData(bytes)), UpstreamUrl = "https://example.org/runtime" };
            var entry = ModelCatalog.Default.Get("Qwen/Qwen3-ASR-0.6B") with { ConvertedFiles = [file] };
            File.WriteAllBytes(Path.Combine(root, file.Name), bytes);
            QwenPrecisionAssets.EnsureBFloat16(entry, root, TestContext.Current.CancellationToken);
            using var canceled = new CancellationTokenSource();
            canceled.Cancel();
            Assert.Throws<OperationCanceledException>(() => QwenPrecisionAssets.EnsureBFloat16(entry, root, canceled.Token));
        }
        finally { Directory.Delete(root, true); }
    }

    private sealed class RejectingSource : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => throw new ModelNetworkException("Source reached");
    }
    private sealed class BytesSource(byte[] bytes) : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => ValueTask.FromResult<Stream>(new MemoryStream(bytes));
    }
}
