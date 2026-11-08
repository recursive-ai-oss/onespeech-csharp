using System.Collections.Immutable;
using System.Text.Json;
using OneSpeech.Serialization;

namespace OneSpeech.Catalog;

/// <summary>A pinned catalog file; URLs are metadata, not authority to bypass tier or gate rules.</summary>
public sealed record CatalogFile
{
    /// <summary>Relative model filename, never an absolute path.</summary>
    public required string Name { get; init; }
    /// <summary>Expected byte size, nonnegative.</summary>
    public required long Size { get; init; }
    /// <summary>Expected SHA-256, 64 hexadecimal digits.</summary>
    public required string Sha256 { get; init; }
    /// <summary>Pinned publisher HTTPS URL.</summary>
    public required string UpstreamUrl { get; init; }
    /// <summary>Optional tier-1 mirror HTTPS URL; null for upstream-only files.</summary>
    public string? MirrorUrl { get; init; }
}

/// <summary>Data-only catalog entry. Download authorization and tier enforcement belong to the loader.</summary>
public sealed record CatalogEntry
{
    /// <summary>Stable catalog identifier.</summary>
    public required string Id { get; init; }
    /// <summary>Explicit family registration key.</summary>
    public required string Family { get; init; }
    /// <summary>Upstream repository URL.</summary>
    public required string Source { get; init; }
    /// <summary>Pinned upstream revision.</summary>
    public required string Revision { get; init; }
    /// <summary>Files required to load this model.</summary>
    public ImmutableArray<CatalogFile> Files { get; init; } = [];
    /// <summary>Converted runtime files; empty while conversion metadata is pending. Never populated for tier 2.</summary>
    public ImmutableArray<CatalogFile> ConvertedFiles { get; init; } = [];
    /// <summary>Whether this entry is the measured family default; false unless explicitly declared. Tier 2 cannot be a default.</summary>
    public bool IsDefault { get; init; }
    /// <summary>Distribution tier.</summary>
    public required ModelTier Tier { get; init; }
    /// <summary>SPDX licence identifier.</summary>
    public required string License { get; init; }
    /// <summary>Exact required attribution text.</summary>
    public required string Attribution { get; init; }
    /// <summary>Whether publisher gate acceptance is required.</summary>
    public bool Gated { get; init; }
    /// <summary>Declared support; no capabilities by default.</summary>
    public Capabilities Capabilities { get; init; } = new();
    /// <summary>Reads bounded catalog-entry UTF-8 JSON without reflection. Throws JsonException for invalid or over-limit input. Does not authorize downloads.</summary>
    public static CatalogEntry FromJson(ReadOnlySpan<byte> utf8Json)
    {
        JsonBounds.Check(utf8Json);
        return JsonSerializer.Deserialize(utf8Json, SpeechJsonContext.Default.CatalogEntry) ?? throw new JsonException("Expected catalog entry.");
    }
}

/// <summary>Host hook for obtaining pinned model bytes from a custom store. The loader must still verify size, digest and tier authorization.</summary>
public abstract class ModelSource
{
    /// <summary>Opens a readable file stream at offset zero. Caller disposes it. Throws ModelNotFoundException if absent and OperationCanceledException on cancellation.</summary>
    public abstract ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken);
    /// <summary>Opens pinned bytes with an ephemeral publisher token. The default delegates to the tokenless host hook. Implementations must never persist or log tokens.</summary>
    public virtual ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, string? token, CancellationToken cancellationToken) => OpenReadAsync(model, file, cancellationToken);
}
