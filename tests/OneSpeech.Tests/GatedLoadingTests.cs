using Xunit;
using System.Net;
using System.Security.Cryptography;
using OneSpeech.Catalog;
using OneSpeech.Diarization;

namespace OneSpeech.Tests;

[Collection("CredentialEnvironment")]
public sealed class GatedLoadingTests
{
    private const string Credential = "synthetic-secret-for-transport-test";
    private static CatalogEntry Entry => new()
    {
        Id = "publisher/gated", Family = "test", Source = "https://huggingface.co/publisher/gated",
        Revision = new string('a', 40), Tier = ModelTier.Tier2, Gated = true, License = "MIT", Attribution = "Synthetic test",
        Files = [new() { Name = "weights", Size = 3, Sha256 = System.Convert.ToHexStringLower(SHA256.HashData(new byte[] { 1, 2, 3 })), UpstreamUrl = "https://huggingface.co/publisher/gated/resolve/" + new string('a', 40) + "/weights" }]
    };

    [Fact]
    public async Task MissingTokenNeverReachesTransport()
    {
        using var client = new HttpClient(new Handler(_ => throw new InvalidOperationException("Must not send")));
        var source = new HttpModelSource(client);
        await Assert.ThrowsAsync<ModelTokenRequiredException>(async () => await source.OpenReadAsync(Entry, Entry.Files[0], null, TestContext.Current.CancellationToken));
        Assert.Throws<ModelTokenRequiredException>(() => PyannoteSegmentation.Load(token: "", cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<FeatureUnavailableException>(() => PyannoteSegmentation.Load(token: Credential, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData(401, false)]
    [InlineData(403, false)]
    [InlineData(403, true)]
    public async Task PublisherRefusalsAreTypedAndSanitized(int status, bool gated)
    {
        using var client = new HttpClient(new Handler(_ =>
        {
            var response = new HttpResponseMessage((HttpStatusCode)status) { Content = new StringContent(Credential) };
            if (gated) response.Headers.Add("X-Error-Code", "GatedRepo");
            response.Headers.Add("X-Error-Message", Credential);
            return response;
        }));
        var source = new HttpModelSource(client);
        var exception = await Assert.ThrowsAnyAsync<OneSpeechException>(async () => await source.OpenReadAsync(Entry, Entry.Files[0], Credential, TestContext.Current.CancellationToken));
        if (gated) Assert.IsType<ModelGateNotAcceptedException>(exception); else Assert.IsType<ModelTokenRejectedException>(exception);
        Assert.DoesNotContain(Credential, exception.ToString());
    }

    [Theory]
    [InlineData("https://cdn.example/weights")]
    [InlineData("https://huggingface.co/next")]
    public async Task RedirectsDoNotCarryCredentials(string target)
    {
        int calls = 0;
        using var client = new HttpClient(new Handler(request =>
        {
            if (calls++ == 0)
            {
                Assert.Equal("https://huggingface.co", request.RequestUri!.GetLeftPart(UriPartial.Authority));
                Assert.Equal(Credential, request.Headers.Authorization?.Parameter);
                var response = new HttpResponseMessage(HttpStatusCode.Redirect);
                response.Headers.Location = new Uri(target); return response;
            }
            Assert.Null(request.Headers.Authorization);
            return new(HttpStatusCode.OK) { Content = new ByteArrayContent([1, 2, 3]) };
        }));
        using var stream = await new HttpModelSource(client).OpenReadAsync(Entry, Entry.Files[0], Credential, TestContext.Current.CancellationToken);
        Assert.Equal(1, stream.ReadByte()); Assert.Equal(2, calls);
    }

    [Fact]
    public async Task StorePersistsOnlyVerifiedBytesAndPublicProvenance()
    {
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        try
        {
            using var client = new HttpClient(new Handler(request =>
            {
                Assert.Equal(Credential, request.Headers.Authorization?.Parameter);
                return new(HttpStatusCode.OK) { Content = new ByteArrayContent([1, 2, 3]) };
            }));
            var model = await new ModelStore(root, new HttpModelSource(client), new([Entry])).PullAsync(Entry.Id, Credential, cancellationToken: TestContext.Current.CancellationToken);
            Assert.DoesNotContain(Credential, model.Info.ToString());
            foreach (string path in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
                Assert.DoesNotContain(Credential, await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public async Task EnvironmentTokenReachesPublisherAndExplicitArgumentWins()
    {
        string? previous = Environment.GetEnvironmentVariable("ONESPEECH_MODEL_TOKEN");
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        try
        {
            Environment.SetEnvironmentVariable("ONESPEECH_MODEL_TOKEN", Credential);
            string expected = Credential;
            using var client = new HttpClient(new Handler(request =>
            {
                Assert.Equal(expected, request.Headers.Authorization?.Parameter);
                return new(HttpStatusCode.OK) { Content = new ByteArrayContent([1, 2, 3]) };
            }));
            var source = new HttpModelSource(client);
            await new ModelStore(Path.Combine(root, "env"), source, new([Entry])).PullAsync(Entry.Id, cancellationToken: TestContext.Current.CancellationToken);
            expected = "explicit-synthetic-credential";
            await new ModelStore(Path.Combine(root, "api"), source, new([Entry])).PullAsync(Entry.Id, expected, cancellationToken: TestContext.Current.CancellationToken);
            foreach (string path in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
            {
                string text = await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken);
                Assert.DoesNotContain(Credential, text); Assert.DoesNotContain(expected, text);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("ONESPEECH_MODEL_TOKEN", previous);
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    [Fact]
    public async Task TransportExceptionsAndMalformedCredentialsAreSanitized()
    {
        using var client = new HttpClient(new Handler(_ => throw new HttpRequestException(Credential)));
        var error = await Assert.ThrowsAsync<ModelNetworkException>(async () => await new HttpModelSource(client).OpenReadAsync(Entry, Entry.Files[0], Credential, TestContext.Current.CancellationToken));
        Assert.DoesNotContain(Credential, error.ToString());
        var invalid = await Assert.ThrowsAsync<ModelTokenRequiredException>(async () => await new HttpModelSource(client).OpenReadAsync(Entry, Entry.Files[0], Credential + "\r\n", TestContext.Current.CancellationToken));
        Assert.DoesNotContain(Credential, invalid.ToString());
    }

    [Fact]
    public async Task MirrorDoesNotReceivePublisherCredential()
    {
        var file = Entry.Files[0] with { MirrorUrl = "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/fixture/weights" };
        var entry = Entry with { Gated = false, Tier = ModelTier.Tier1, Files = [file] };
        using var client = new HttpClient(new Handler(request =>
        {
            Assert.Equal("github.com", request.RequestUri!.Host); Assert.Null(request.Headers.Authorization);
            return new(HttpStatusCode.OK) { Content = new ByteArrayContent([1, 2, 3]) };
        }));
        using var stream = await new HttpModelSource(client).OpenReadAsync(entry, file, Credential, TestContext.Current.CancellationToken);
        Assert.Equal(1, stream.ReadByte());
    }

    [Fact]
    public async Task SynchronousAndAsyncStreamFailuresDoNotExposeResponseDetails()
    {
        using var client = new HttpClient(new Handler(_ => new(HttpStatusCode.OK) { Content = new StreamContent(new FailingStream()) }));
        using var stream = await new HttpModelSource(client).OpenReadAsync(Entry, Entry.Files[0], Credential, TestContext.Current.CancellationToken);
        var syncError = Assert.Throws<ModelNetworkException>(() => stream.ReadByte());
        Assert.DoesNotContain(Credential, syncError.ToString());
        var asyncError = await Assert.ThrowsAsync<ModelNetworkException>(async () => await stream.ReadExactlyAsync(new byte[1], TestContext.Current.CancellationToken));
        Assert.DoesNotContain(Credential, asyncError.ToString());
    }

    private sealed class FailingStream() : MemoryStream(new byte[3])
    {
        public override int Read(byte[] buffer, int offset, int count) => throw new IOException(Credential);
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => throw new IOException(Credential);
    }

    private sealed class Handler(Func<HttpRequestMessage, HttpResponseMessage> send) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => Task.FromResult(send(request));
    }
}

[CollectionDefinition("CredentialEnvironment", DisableParallelization = true)]
public sealed class CredentialEnvironmentTests;
