using System.Net;
using System.Net.Http.Headers;

namespace OneSpeech.Catalog;

/// <summary>A publisher or mirror request failed. Credentials and signed URLs are not included in the error.</summary>
public sealed class ModelNetworkException : OneSpeechException
{
    /// <summary>Creates a sanitized transport failure.</summary>
    public ModelNetworkException(string message) : base(message) { }
}

/// <summary>HTTPS transport with at most five HTTPS redirects. Tokens go only to the original repository endpoint, never to redirect targets.</summary>
public sealed class HttpModelSource : ModelSource
{
    private static readonly HttpClient Client = new(new HttpClientHandler { AllowAutoRedirect = false }) { Timeout = TimeSpan.FromMinutes(30) };

    /// <inheritdoc />
    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => OpenReadAsync(model, file, null, cancellationToken);

    /// <inheritdoc />
    public override async ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, string? token, CancellationToken cancellationToken)
    {
        ModelCatalog.Validate(model);
        if (!model.Files.Contains(file) && !model.ConvertedFiles.Contains(file)) throw new InvalidModelCatalogException("File is not part of this model.");
        if (model.Gated && string.IsNullOrWhiteSpace(token)) throw new ModelTokenRequiredException("Set ONESPEECH_MODEL_TOKEN or pass a token after accepting the publisher's gate.");
        // Tier 2 never reaches a mirror, including when this source is used outside ModelStore.
        var uri = ModelCatalog.Https(model.Tier == ModelTier.Tier1 ? file.MirrorUrl ?? file.UpstreamUrl : file.UpstreamUrl);
        try
        {
            for (int redirects = 0; redirects <= 5; redirects++)
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, uri);
                if (redirects == 0 && token is not null && uri.AbsoluteUri == file.UpstreamUrl)
                    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                var response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                if (response.StatusCode is HttpStatusCode.Moved or HttpStatusCode.Redirect or HttpStatusCode.RedirectMethod or HttpStatusCode.TemporaryRedirect or HttpStatusCode.PermanentRedirect)
                {
                    var location = response.Headers.Location;
                    response.Dispose();
                    if (location is null || redirects == 5) throw new ModelNetworkException("Invalid or excessive model redirects.");
                    uri = ModelCatalog.Https(new Uri(uri, location).AbsoluteUri);
                    continue;
                }
                if (!response.IsSuccessStatusCode)
                {
                    var status = response.StatusCode;
                    response.Dispose();
                    if (status == HttpStatusCode.NotFound) throw new ModelNotFoundException($"Publisher file not found for '{model.Id}'.");
                    if (status is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden) throw new ModelTokenRequiredException("Publisher refused access. Accept its gate and supply ONESPEECH_MODEL_TOKEN or an API token.");
                    throw new ModelNetworkException($"Model request failed (HTTP {(int)status}).");
                }
                if (response.Content.Headers.ContentLength is long length && length != file.Size)
                {
                    response.Dispose();
                    throw new ModelIntegrityException($"Unexpected Content-Length for '{file.Name}'.");
                }
                try { return new ResponseStream(await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false), response); }
                catch { response.Dispose(); throw; }
            }
        }
        catch (HttpRequestException) { throw new ModelNetworkException("Model network request failed; check connectivity and retry."); }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { throw new ModelNetworkException("Model request timed out; retry the pull."); }
        throw new ModelNetworkException("Model redirect limit reached.");
    }

    private sealed class ResponseStream(Stream inner, HttpResponseMessage response) : Stream
    {
        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            try { return await inner.ReadAsync(buffer, cancellationToken).ConfigureAwait(false); }
            catch (IOException) { throw new ModelNetworkException("Model response stream failed; retry the pull."); }
            catch (HttpRequestException) { throw new ModelNetworkException("Model response stream failed; retry the pull."); }
        }
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { if (disposing) { inner.Dispose(); response.Dispose(); } base.Dispose(disposing); }
    }
}

/// <summary>Reads verified publisher bytes from a host-owned directory, without network. Files are beneath root/model-id/revision/name. The root must be trusted and contain no symbolic links.</summary>
public sealed class DirectoryModelSource : ModelSource
{
    private readonly string root;
    /// <summary>Uses an existing or future trusted local root. Throws ArgumentException for an invalid path.</summary>
    public DirectoryModelSource(string directory) => root = Path.GetFullPath(directory);
    /// <inheritdoc />
    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken)
    {
        ModelCatalog.Validate(model);
        if (!model.Files.Contains(file) && !model.ConvertedFiles.Contains(file)) throw new InvalidModelCatalogException("File is not part of this model.");
        cancellationToken.ThrowIfCancellationRequested();
        string path = Path.Combine(root, model.Id, model.Revision, file.Name);
        try { return ValueTask.FromResult<Stream>(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous | FileOptions.SequentialScan)); }
        catch (FileNotFoundException) { throw new ModelNotFoundException($"Local file '{file.Name}' for '{model.Id}' is missing."); }
        catch (DirectoryNotFoundException) { throw new ModelNotFoundException($"Local model '{model.Id}' is missing."); }
    }
}
