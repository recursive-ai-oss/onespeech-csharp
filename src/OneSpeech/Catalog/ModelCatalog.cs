using System.Collections.Immutable;
using System.Text.Json;
using System.Text.Json.Serialization;
using OneSpeech.Serialization;

namespace OneSpeech.Catalog;

/// <summary>Catalog JSON is malformed, unsupported, or violates download policy.</summary>
public sealed class InvalidModelCatalogException : OneSpeechException
{
    /// <summary>Creates a catalog validation failure.</summary>
    public InvalidModelCatalogException(string message) : base(message) { }
    /// <summary>Creates a catalog validation failure preserving the parsing cause.</summary>
    public InvalidModelCatalogException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>A bounded, version-1 catalog of pinned files. Immutable after validation.</summary>
public sealed class ModelCatalog
{
    /// <summary>The built-in catalog. No I/O other than reading the embedded resource; never downloads.</summary>
    public static ModelCatalog Default { get; } = ReadEmbedded();
    /// <summary>Validated entries in catalog order; at most 256.</summary>
    public ImmutableArray<CatalogEntry> Entries { get; }

    /// <summary>Validates up to 256 entries, each with at most 64 files. Throws InvalidModelCatalogException on policy violations.</summary>
    public ModelCatalog(IEnumerable<CatalogEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);
        var builder = ImmutableArray.CreateBuilder<CatalogEntry>();
        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var entry in entries)
        {
            if (builder.Count == 256) Fail("Too many models.");
            Validate(entry);
            if (!ids.Add(entry.Id)) Fail("Duplicate model ID.");
            builder.Add(entry);
        }
        Entries = builder.ToImmutable();
        if (Entries.Where(e => e.IsDefault).GroupBy(e => e.Family, StringComparer.Ordinal).Any(g => g.Count() > 1)) Fail("Duplicate family default.");
    }

    /// <summary>Finds an exact, case-sensitive ID; throws ModelNotFoundException if absent.</summary>
    public CatalogEntry Get(string id) => Entries.FirstOrDefault(e => e.Id == id)
        ?? throw new ModelNotFoundException($"Unknown model '{id}'. Use 'onespeech models list'.");

    /// <summary>Finds the one declared tier-1 default for a family, or throws ModelNotFoundException. Tier 2 is never eligible.</summary>
    public CatalogEntry GetDefault(string family) => Entries.FirstOrDefault(e => e.Family == family && e.IsDefault && e.Tier == ModelTier.Tier1)
        ?? throw new ModelNotFoundException($"No default is declared for family '{family}'; select a model ID explicitly.");

    /// <summary>Reads UTF-8 JSON, bounded to 8 MiB, depth 32, 256 items per array, and 64 KiB strings before allocation. Throws InvalidModelCatalogException.</summary>
    public static ModelCatalog FromJson(ReadOnlySpan<byte> json)
    {
        try
        {
            JsonBounds.Check(json);
            // Bound every array and reject duplicate properties before the generated reader allocates collections.
            var reader = new Utf8JsonReader(json, new JsonReaderOptions { MaxDepth = 32 });
            var objects = new Stack<HashSet<string>>();
            var arrays = new int[33];
            var isArray = new bool[33];
            var limits = new int[33];
            string? property = null;
            while (reader.Read())
            {
                int depth = reader.CurrentDepth;
                if (reader.TokenType is not (JsonTokenType.EndArray or JsonTokenType.EndObject or JsonTokenType.PropertyName) && depth > 0 && isArray[depth - 1] && ++arrays[depth - 1] > limits[depth - 1])
                    Fail("Too many array items.");
                if (reader.TokenType == JsonTokenType.StartArray)
                {
                    isArray[depth] = true; arrays[depth] = 0;
                    limits[depth] = property is "files" or "convertedFiles" ? 64 : 256;
                }
                if (reader.TokenType == JsonTokenType.EndArray) isArray[depth] = false;
                if (reader.TokenType == JsonTokenType.StartObject) objects.Push(new(StringComparer.Ordinal));
                if (reader.TokenType == JsonTokenType.EndObject) objects.Pop();
                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    property = reader.GetString();
                    if (!objects.Peek().Add(property!)) Fail("Duplicate JSON property.");
                }
            }
            var envelope = JsonSerializer.Deserialize(json, CatalogJsonContext.Default.CatalogEnvelope);
            if (envelope is null || envelope.SchemaVersion != 1 || envelope.Models.IsDefault) Fail("Expected catalog schemaVersion 1 and models array.");
            return new ModelCatalog(envelope!.Models);
        }
        catch (JsonException e) { throw new InvalidModelCatalogException("Invalid catalog JSON.", e); }
    }

    private static ModelCatalog ReadEmbedded()
    {
        using var stream = typeof(ModelCatalog).Assembly.GetManifestResourceStream("OneSpeech.Catalog.models.json")
            ?? throw new InvalidModelCatalogException("Embedded catalog is missing.");
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return FromJson(buffer.GetBuffer().AsSpan(0, (int)buffer.Length));
    }

    internal static void Validate(CatalogEntry? entry)
    {
        if (entry is null || !SafePath(entry.Id) || string.IsNullOrWhiteSpace(entry.Family) ||
            entry.Revision is not { Length: 40 } || !entry.Revision.All(char.IsAsciiHexDigit) ||
            string.IsNullOrWhiteSpace(entry.Attribution) || entry.Capabilities is null ||
            entry.Files.IsDefaultOrEmpty || entry.Files.Length > 64 || entry.Capabilities.Languages.IsDefault || entry.Capabilities.Languages.Length > 256 || entry.ConvertedFiles.IsDefault || entry.ConvertedFiles.Length > 64)
            Fail("Missing or invalid model metadata.");
        var e = entry!;
        if (e.Tier == ModelTier.Tier1)
        {
            if (e.Gated || e.License is not ("MIT" or "Apache-2.0" or "BSD-2-Clause" or "BSD-3-Clause")) Fail("Tier 1 requires an ungated permissive licence.");
        }
        else if (e.Tier != ModelTier.Tier2 || e.IsDefault || e.License is not ("MIT" or "Apache-2.0" or "BSD-2-Clause" or "BSD-3-Clause" or "CC-BY-4.0" or "OpenMDW-1.1"))
            Fail("Unsupported tier, licence, or tier-2 default.");
        if (e.Tier == ModelTier.Tier2 && !e.ConvertedFiles.IsEmpty) Fail("Tier 2 cannot have redistributed converted files.");
        var source = Https(e.Source);
        if (source.Query.Length != 0 || source.Fragment.Length != 0 || source.AbsolutePath.Trim('/').Split('/').Length < 2) Fail("Expected a publisher repository URL.");
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var f in e.Files.Concat(e.ConvertedFiles))
        {
            if (f is null || !SafePath(f.Name) || !names.Add(f.Name) || f.Size < 0 || f.Size > 64L * 1024 * 1024 * 1024 ||
                f.Sha256 is not { Length: 64 } || !f.Sha256.All(char.IsAsciiHexDigit)) Fail("Invalid file metadata.");
            var upstream = Https(f!.UpstreamUrl);
            string repo = source.AbsolutePath.TrimEnd('/');
            bool publisher = upstream.Host == source.Host && upstream.AbsolutePath.StartsWith(repo + "/resolve/" + e.Revision + "/", StringComparison.Ordinal);
            publisher |= source.Host == "github.com" && upstream.Host == "raw.githubusercontent.com" && upstream.AbsolutePath.StartsWith(repo + "/" + e.Revision + "/", StringComparison.Ordinal);
            bool converted = e.ConvertedFiles.Contains(f);
            if (converted) publisher = e.Tier == ModelTier.Tier1 && upstream.Host == "github.com" && upstream.AbsolutePath.StartsWith("/recursive-ai-oss/onespeech-csharp/releases/download/", StringComparison.Ordinal);
            if (!publisher || upstream.Query.Length != 0 || upstream.Fragment.Length != 0) Fail("File must be pinned to the publisher repository revision.");
            if (f.MirrorUrl is not null)
            {
                if (e.Tier != ModelTier.Tier1) Fail("Tier 2 must never have a mirror.");
                var mirror = Https(f.MirrorUrl);
                if (mirror.Host != "github.com" || !mirror.AbsolutePath.StartsWith("/recursive-ai-oss/onespeech-csharp/releases/download/", StringComparison.Ordinal)) Fail("Mirror must be a OneSpeech release asset.");
            }
        }
        foreach (string name in names)
        {
            string[] parts = name.Split('/');
            for (int i = 1; i < parts.Length; i++)
                if (names.Contains(string.Join('/', parts, 0, i))) Fail("A file path is also a directory.");
        }
    }

    internal static bool SafePath(string? value) => value is { Length: > 0 and <= 240 } &&
        value.Split('/').All(p => p.Length > 0 && char.IsAsciiLetterOrDigit(p[0]) && !ReservedName(p) && p is not ("." or "..") && !p.EndsWith('.') &&
            p.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.'));

    private static bool ReservedName(string component)
    {
        string stem = component.Split('.')[0].ToUpperInvariant();
        return stem is "CON" or "PRN" or "AUX" or "NUL" ||
            stem.Length == 4 && (stem.StartsWith("COM", StringComparison.Ordinal) || stem.StartsWith("LPT", StringComparison.Ordinal)) && stem[3] is >= '1' and <= '9';
    }

    internal static Uri Https(string? value)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) || uri.Scheme != "https" || uri.UserInfo.Length != 0 || !uri.IsDefaultPort)
            throw new InvalidModelCatalogException("Expected an HTTPS URL without credentials or custom port.");
        return uri;
    }

    private static void Fail(string message) => throw new InvalidModelCatalogException(message);
}

internal sealed record CatalogEnvelope
{
    public required int SchemaVersion { get; init; }
    public required ImmutableArray<CatalogEntry> Models { get; init; }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, UseStringEnumConverter = true,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
[JsonSerializable(typeof(CatalogEnvelope))]
internal sealed partial class CatalogJsonContext : JsonSerializerContext;
