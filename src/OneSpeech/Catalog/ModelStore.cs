using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text;

namespace OneSpeech.Catalog;

/// <summary>Progress for one file, in bytes; emitted during download, not for cache hits.</summary>
public sealed record ModelDownloadProgress(string ModelId, string FileName, long BytesReceived, long TotalBytes);

/// <summary>A directory of verified publisher files and their immutable provenance. Converted runtime files are a separate catalog artifact.</summary>
public sealed record ResolvedModel(string Directory, ModelInfo Info);

/// <summary>Resolves explicit model IDs to verified local publisher files. Never chooses a tier-2 default and never makes network requests for present files.</summary>
public sealed class ModelStore
{
    private readonly ModelCatalog catalog;
    private readonly ModelSource source;
    // Process-local markers: every new store verifies disk content once, rather than trusting a writable persistent marker.
    private readonly ConcurrentDictionary<string, (long Size, long Ticks)> verified = new(StringComparer.Ordinal);
    /// <summary>Absolute cache root. A trusted per-user directory, not a shared writable store.</summary>
    public string CacheDirectory { get; }

    /// <summary>Creates a store. Cache precedence: argument, ONESPEECH_MODEL_CACHE, platform default. A source override replaces HTTP entirely; catalog defaults to the embedded catalog.</summary>
    public ModelStore(string? cacheDirectory = null, ModelSource? source = null, ModelCatalog? catalog = null)
    {
        CacheDirectory = Path.GetFullPath(cacheDirectory ?? Environment.GetEnvironmentVariable("ONESPEECH_MODEL_CACHE") ?? DefaultCacheDirectory);
        this.source = source ?? new HttpModelSource();
        this.catalog = catalog ?? ModelCatalog.Default;
    }

    /// <summary>Default cache: LocalApplicationData/OneSpeech/models on Windows; ~/Library/Caches/OneSpeech/models on macOS; XDG_CACHE_HOME/onespeech/models (or ~/.cache/onespeech/models) elsewhere.</summary>
    public static string DefaultCacheDirectory => OperatingSystem.IsWindows()
        ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "OneSpeech", "models")
        : OperatingSystem.IsMacOS()
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Library", "Caches", "OneSpeech", "models")
            : Path.Combine(Environment.GetEnvironmentVariable("XDG_CACHE_HOME") ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache"), "onespeech", "models");

    /// <summary>Downloads missing publisher files and verifies present files on first open per store. Restartable, atomic per file, with an OS file lock across processes. A corrupt present file is deleted and reported, not silently replaced. Gate tokens come from the argument or ONESPEECH_MODEL_TOKEN and are never persisted. Throws typed model errors or OperationCanceledException. Progress counts bytes per file.</summary>
    public async ValueTask<ResolvedModel> PullAsync(string id, string? token = null, IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var entry = catalog.Get(id);
        token ??= Environment.GetEnvironmentVariable("ONESPEECH_MODEL_TOKEN");
        if (entry.Gated && string.IsNullOrWhiteSpace(token)) throw new ModelTokenRequiredException("Accept the publisher gate and pass a token or set ONESPEECH_MODEL_TOKEN.");
        string key = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(entry.Id + "\n" + entry.Revision + "\n" + string.Join('\n', entry.Files.Select(f => f.Name + ":" + f.Sha256)))));
        string directory = Path.Combine(CacheDirectory, key);
        System.IO.Directory.CreateDirectory(directory);
        using var lease = await LockAsync(Path.Combine(directory, ".lock"), cancellationToken).ConfigureAwait(false);
        foreach (var file in entry.Files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string path = Path.Combine(directory, file.Name);
            if (File.Exists(path))
            {
                var info = new FileInfo(path);
                var stamp = (info.Length, info.LastWriteTimeUtc.Ticks);
                if (verified.TryGetValue(path, out var marker) && marker == stamp) continue;
                if (info.Length != file.Size || !await MatchesAsync(path, file.Sha256, cancellationToken).ConfigureAwait(false))
                {
                    File.Delete(path);
                    verified.TryRemove(path, out _);
                    throw new ModelIntegrityException($"Cached file '{file.Name}' failed verification; removed it. Retry the pull.");
                }
                verified[path] = stamp;
                continue;
            }
            System.IO.Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            string temporary = Path.Combine(directory, ".download");
            try
            {
                using var input = await source.OpenReadAsync(entry, file, token, cancellationToken).ConfigureAwait(false);
                await using (var output = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None, 65536, FileOptions.Asynchronous | FileOptions.SequentialScan))
                {
                    using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                    byte[] buffer = new byte[65536];
                    long total = 0;
                    while (true)
                    {
                        int count = await input.ReadAsync(buffer.AsMemory(0, (int)Math.Min(buffer.Length, file.Size - total + 1)), cancellationToken).ConfigureAwait(false);
                        if (count == 0) break;
                        total += count;
                        if (total > file.Size) throw new ModelIntegrityException($"File '{file.Name}' exceeds its catalog size.");
                        hash.AppendData(buffer, 0, count);
                        await output.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
                        progress?.Report(new(entry.Id, file.Name, total, file.Size));
                    }
                    if (total != file.Size || !Convert.ToHexStringLower(hash.GetHashAndReset()).Equals(file.Sha256, StringComparison.OrdinalIgnoreCase))
                        throw new ModelIntegrityException($"File '{file.Name}' failed size or SHA-256 verification.");
                    await output.FlushAsync(cancellationToken).ConfigureAwait(false);
                }
                cancellationToken.ThrowIfCancellationRequested();
                File.Move(temporary, path);
                var info = new FileInfo(path);
                verified[path] = (info.Length, info.LastWriteTimeUtc.Ticks);
            }
            catch (HttpRequestException) { throw new ModelNetworkException("Model transfer failed; retry the pull."); }
            finally { File.Delete(temporary); }
        }
        return new(directory, new ModelInfo
        {
            Id = entry.Id, Source = entry.Source, Revision = entry.Revision, Tier = entry.Tier,
            License = entry.License, Attribution = entry.Attribution, Gated = entry.Gated,
            Files = entry.Files.Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha256 }).ToImmutableArray()
        });
    }

    private static async ValueTask<bool> MatchesAsync(string path, string expected, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false)).Equals(expected, StringComparison.OrdinalIgnoreCase);
    }

    private static async ValueTask<FileStream> LockAsync(string path, CancellationToken cancellationToken)
    {
        for (int attempt = 0; ; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try { return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); }
            catch (IOException) when (attempt < 12000) { await Task.Delay(50, cancellationToken).ConfigureAwait(false); }
        }
    }
}
