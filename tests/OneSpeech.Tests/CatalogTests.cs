using System.Security.Cryptography;
using System.Text;
using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class CatalogTests
{
    private static readonly byte[] Bytes = [1, 2, 3];
    private static string Temp() => Path.Combine(Path.GetTempPath(), "onespeech-catalog-" + Guid.NewGuid().ToString("N"));
    private static CatalogEntry Entry => new()
    {
        Id = "publisher/model", Family = "test", Source = "https://huggingface.co/publisher/model",
        Revision = new string('a', 40), License = "MIT", Tier = ModelTier.Tier1, Attribution = "Test fixture",
        Files = [new() { Name = "weights", Size = 3, Sha256 = Convert.ToHexStringLower(SHA256.HashData(Bytes)), UpstreamUrl = "https://huggingface.co/publisher/model/resolve/" + new string('a', 40) + "/weights" }]
    };

    [Fact]
    public void EmbeddedCatalogHasPinnedCompleteEntriesAndNoUnmeasuredDefault()
    {
        var catalog = ModelCatalog.Default;
        Assert.Equal(9, catalog.Entries.Length);
        Assert.All(catalog.Entries, e => { Assert.Equal(ModelTier.Tier1, e.Tier); Assert.False(e.Gated); Assert.All(e.Files, f => Assert.Null(f.MirrorUrl)); });
        Assert.Throws<ModelNotFoundException>(() => catalog.GetDefault("whisper"));
        Assert.Equal("169d4a4341b33bc18d8881c4b69c2e104e1cc0af", catalog.Get("openai/whisper-tiny").Revision);
    }

    [Fact]
    public void ModelScopeQueriesArePinnedToExactRepoRevisionAndFile()
    {
        var original = Entry;
        var entry = original with { Source = "https://modelscope.cn/models/publisher/model", Files = [original.Files[0] with { UpstreamUrl = "https://modelscope.cn/api/v1/models/publisher/model/repo?Revision=" + original.Revision + "&FilePath=weights" }] };
        Assert.Single(new ModelCatalog([entry]).Entries);
        foreach (string suffix in new[] { "&other=1", "&Revision=master", "#fragment" })
            Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([entry with { Files = [entry.Files[0] with { UpstreamUrl = entry.Files[0].UpstreamUrl + suffix }] }]));
        foreach (var url in new[] { entry.Files[0].UpstreamUrl.Replace(original.Revision, "master", StringComparison.Ordinal), entry.Files[0].UpstreamUrl.Replace("FilePath=weights", "FilePath=other", StringComparison.Ordinal), entry.Files[0].UpstreamUrl.Replace("publisher/model/repo", "publisher/other/repo", StringComparison.Ordinal), entry.Files[0].UpstreamUrl.Replace("modelscope.cn", "example.org", StringComparison.Ordinal) })
            Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([entry with { Files = [entry.Files[0] with { UpstreamUrl = url }] }]));
        var converted = ModelCatalog.Default.Get("iic/speech_campplus_sv_en_voxceleb_16k").ConvertedFiles;
        Assert.Equal("f333a00d5d66b9bf3d7e7abf35f36ff4a01e6e8a0df235b89a125e9fbe3cc15c", Assert.Single(converted).Sha256);
    }

    private const string Valid = """
        {"schemaVersion":1,"models":[{"id":"test/model","family":"test","source":"https://huggingface.co/test/model","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","tier":"Tier1","license":"MIT","attribution":"fixture","convertedFiles":[],"capabilities":{"languages":[]},"files":[{"name":"weights","size":3,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","upstreamUrl":"https://huggingface.co/test/model/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/weights"}]}]}
        """;

    [Theory]
    [InlineData("")]
    [InlineData("{")]
    [InlineData("null")]
    [InlineData("{}")]
    [InlineData("{\"schemaVersion\":2,\"models\":[]}")]
    [InlineData("{\"schemaVersion\":1,\"schemaVersion\":1,\"models\":[]}")]
    [InlineData("{\"schemaVersion\":1,\"models\":[null]}")]
    public void RejectsMalformed(string json) => Assert.Throws<InvalidModelCatalogException>(() => ModelCatalog.FromJson(Encoding.UTF8.GetBytes(json)));

    [Fact]
    public void BoundsAndHostileCatalogsAreRejected()
    {
        Assert.Single(ModelCatalog.FromJson(Encoding.UTF8.GetBytes(Valid)).Entries);
        string duplicate = Valid[..^2] + "," + Valid[(Valid.IndexOf('[') + 1)..^2] + "]}";
        Assert.Throws<InvalidModelCatalogException>(() => ModelCatalog.FromJson(Encoding.UTF8.GetBytes(duplicate)));
        foreach (string hostile in new[] { Valid[..^1], Valid.Replace("\"Tier1\"", "\"Tier2\",\"isDefault\":true", StringComparison.Ordinal), Valid.Replace("\"weights\"", "\"../escape\"", StringComparison.Ordinal), Valid.Replace("\"size\":3", "\"size\":-1", StringComparison.Ordinal), Valid.Replace("\"MIT\"", "\"research-only\"", StringComparison.Ordinal), Valid.Replace("\"attribution\":\"fixture\"", "\"attribution\":null", StringComparison.Ordinal), Valid.Replace("https:", "http:", StringComparison.Ordinal) })
            Assert.Throws<InvalidModelCatalogException>(() => ModelCatalog.FromJson(Encoding.UTF8.GetBytes(hostile)));
        Assert.Throws<InvalidModelCatalogException>(() => ModelCatalog.FromJson(new byte[8 * 1024 * 1024 + 1]));
        Assert.Throws<InvalidModelCatalogException>(() => ModelCatalog.FromJson(Encoding.UTF8.GetBytes("{\"schemaVersion\":1,\"models\":[" + string.Join(',', Enumerable.Repeat("{}", 257)) + "]}")));
        Assert.Throws<InvalidModelCatalogException>(() => ModelCatalog.FromJson([123, 34, 255, 34, 58, 48, 125]));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([Entry, Entry]));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([Entry with { Files = [Entry.Files[0] with { Name = ".lock" }] }]));
    }

    [Fact]
    public async Task SourceOverrideAndCacheAvoidNetworkAndVerifyOnFirstOpen()
    {
        string root = Temp();
        try
        {
            var source = new FakeSource();
            var catalog = new ModelCatalog([Entry]);
            var store = new ModelStore(root, source, catalog);
            var first = await store.PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            var second = await store.PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(first.Directory, second.Directory);
            Assert.Equal(1, source.Calls);
            Assert.Equal(Bytes, await File.ReadAllBytesAsync(Path.Combine(first.Directory, "weights"), TestContext.Current.CancellationToken));
            Assert.Equal(Entry.Revision, first.Info.Revision);
            await new ModelStore(root, new FailingSource(), catalog).PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(Path.Combine(first.Directory, "weights"), [3, 2, 1], TestContext.Current.CancellationToken);
            await Assert.ThrowsAsync<ModelIntegrityException>(async () => await new ModelStore(root, new FailingSource(), catalog).PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken));
            Assert.False(File.Exists(Path.Combine(first.Directory, "weights")));
        }
        finally { Directory.Delete(root, true); }
    }

    [Fact]
    public async Task WrongHashAndOversizeNeverCommitBadFiles()
    {
        foreach (byte[] bytes in new byte[][] { [3, 2, 1], [1, 2, 3, 4], [1] })
        {
            string root = Temp();
            try
            {
                var store = new ModelStore(root, new FakeSource(bytes), new([Entry]));
                await Assert.ThrowsAsync<ModelIntegrityException>(async () => await store.PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken));
                Assert.Empty(Directory.GetFiles(root, "weights", SearchOption.AllDirectories));
                Assert.Empty(Directory.GetFiles(root, ".download", SearchOption.AllDirectories));
            }
            finally { Directory.Delete(root, true); }
        }
    }

    [Fact]
    public async Task ConcurrentStoresUseProcessLevelLockAndOneGoodCopy()
    {
        string root = Temp();
        try
        {
            var source = new FakeSource(delay: true);
            var catalog = new ModelCatalog([Entry]);
            var results = await Task.WhenAll(new ModelStore(root, source, catalog).PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken).AsTask(), new ModelStore(root, source, catalog).PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken).AsTask());
            Assert.Equal(results[0].Directory, results[1].Directory);
            Assert.Equal(1, source.Calls);
            Assert.Equal(Bytes, await File.ReadAllBytesAsync(Path.Combine(results[0].Directory, "weights"), TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(root, true); }
    }

    [Fact]
    public async Task CancelMidTransferLeavesNoVisiblePartial()
    {
        string root = Temp();
        try
        {
            using var cancellation = new CancellationTokenSource();
            var source = new CancellingSource(cancellation);
            await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await new ModelStore(root, source, new([Entry])).PullAsync(Entry.Id, cancellationToken: cancellation.Token));
            Assert.Empty(Directory.GetFiles(root, "weights", SearchOption.AllDirectories));
            Assert.Empty(Directory.GetFiles(root, ".download", SearchOption.AllDirectories));
        }
        finally { Directory.Delete(root, true); }
    }

    [Fact]
    public async Task Tier2AndGatesAreEnforcedBeforeCustomSource()
    {
        var tier2 = Entry with { Tier = ModelTier.Tier2, License = "CC-BY-4.0" };
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([tier2 with { IsDefault = true }]));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([tier2 with { Files = [tier2.Files[0] with { MirrorUrl = "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/test/weights" }] }]));
        var catalog = new ModelCatalog([tier2]);
        Assert.Throws<ModelNotFoundException>(() => catalog.GetDefault("test"));
        string root = Temp();
        try
        {
            var source = new FakeSource();
            var gated = tier2 with { Gated = true };
            // A whitespace token explicitly overrides any ambient credentials.
            await Assert.ThrowsAsync<ModelTokenRequiredException>(async () => await new ModelStore(root, source, new([gated])).PullAsync(gated.Id, token: " ", cancellationToken: TestContext.Current.CancellationToken));
            Assert.Equal(0, source.Calls);
            await new ModelStore(root, source, catalog).PullAsync(tier2.Id, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(1, source.Calls);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public async Task DirectorySourceUsesPublisherLayoutAndUnknownModelIsTyped()
    {
        string root = Temp();
        string cache = Temp();
        try
        {
            string path = Path.Combine(root, Entry.Id, Entry.Revision);
            Directory.CreateDirectory(path);
            await File.WriteAllBytesAsync(Path.Combine(path, "weights"), Bytes, TestContext.Current.CancellationToken);
            var store = new ModelStore(cache, new DirectoryModelSource(root), new([Entry]));
            var result = await store.PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(Bytes, await File.ReadAllBytesAsync(Path.Combine(result.Directory, "weights"), TestContext.Current.CancellationToken));
            await Assert.ThrowsAsync<ModelNotFoundException>(async () => await store.PullAsync("unknown", cancellationToken: TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(root, true); if (Directory.Exists(cache)) Directory.Delete(cache, true); }
    }

    [Fact]
    public async Task InvalidTokenIsRejectedWithoutEchoingItOrSendingARequest()
    {
        var entry = Entry with { Tier = ModelTier.Tier2, Gated = true };
        var error = await Assert.ThrowsAsync<ModelTokenRequiredException>(async () =>
            await new HttpModelSource().OpenReadAsync(entry, entry.Files[0], "sensitive\r\nvalue", TestContext.Current.CancellationToken));
        Assert.DoesNotContain("sensitive", error.ToString(), StringComparison.Ordinal);
    }

    private sealed class FakeSource(byte[]? bytes = null, bool delay = false) : ModelSource
    {
        internal int Calls;
        public override async ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref Calls);
            if (delay) await Task.Delay(100, cancellationToken);
            return new MemoryStream(bytes ?? Bytes);
        }
    }
    private sealed class FailingSource : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => throw new InvalidOperationException("Network must not be used.");
    }
    private sealed class CancellingSource(CancellationTokenSource cancellation) : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => ValueTask.FromResult<Stream>(new CancellingStream(cancellation));
    }
    private sealed class CancellingStream(CancellationTokenSource cancellation) : MemoryStream(Bytes)
    {
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellation.Cancel();
            return base.ReadAsync(buffer[..1], cancellationToken);
        }
    }
}
