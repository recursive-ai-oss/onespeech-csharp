using System.Collections.Immutable;

namespace OneSpeech;

/// <summary>Weight distribution class. Tier is not inferred from the licence identifier.</summary>
public enum ModelTier
{
    /// <summary>Ungated permissive weights eligible for defaults and redistribution.</summary>
    Tier1 = 1,
    /// <summary>Upstream-only weights, never a default or redistributed.</summary>
    Tier2 = 2
}

/// <summary>A weight file's immutable provenance.</summary>
public sealed record ModelFileInfo
{
    /// <summary>Relative upstream filename.</summary>
    public required string Name { get; init; }
    /// <summary>SHA-256 as 64 hexadecimal digits.</summary>
    public required string Sha256 { get; init; }
}

/// <summary>Model provenance carried with recognition output.</summary>
public sealed record ModelInfo
{
    /// <summary>Catalog model identifier.</summary>
    public required string Id { get; init; }
    /// <summary>Upstream repository URL.</summary>
    public required string Source { get; init; }
    /// <summary>Pinned upstream revision, not a moving branch.</summary>
    public required string Revision { get; init; }
    /// <summary>Digests of all weight files used.</summary>
    public ImmutableArray<ModelFileInfo> Files { get; init; } = [];
    /// <summary>Distribution tier.</summary>
    public required ModelTier Tier { get; init; }
    /// <summary>SPDX licence identifier.</summary>
    public required string License { get; init; }
    /// <summary>Exact required attribution text, preserved without normalization.</summary>
    public required string Attribution { get; init; }
    /// <summary>Whether the publisher requires gate acceptance; false by default.</summary>
    public bool Gated { get; init; }
}

/// <summary>Declared model support. Unsupported requests must be refused, not silently ignored.</summary>
public sealed record Capabilities
{
    /// <summary>Supported language tags; empty means none declared, not all languages.</summary>
    public ImmutableArray<string> Languages { get; init; } = [];
    /// <summary>Whether translation is supported; false by default.</summary>
    public bool Translation { get; init; }
    /// <summary>Whether native word timestamps are supported; false by default.</summary>
    public bool NativeTimestamps { get; init; }
    /// <summary>Whether stateful live streaming is supported; false by default. File chunking alone does not imply this.</summary>
    public bool Streaming { get; init; }
    /// <summary>Whether contextual phrase biasing is supported; false by default.</summary>
    public bool Biasing { get; init; }
    /// <summary>Maximum simultaneous speakers, or null when no limit is declared.</summary>
    public int? MaxSpeakers { get; init; }
}
