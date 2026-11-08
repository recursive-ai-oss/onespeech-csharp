using System.Text.RegularExpressions;
using OneSpeech;

string root = Path.GetFullPath(args.Length == 1 ? args[0] : ".");
string compiled = File.ReadAllText(Path.Combine(root, "tests", "DocsSamples", "Examples.cs"));
int snippets = 0;
int links = 0;
var files = Directory.GetFiles(Path.Combine(root, "docs"), "*.md", SearchOption.AllDirectories)
    .Append(Path.Combine(root, "README.md"));
foreach (string file in files)
{
    string markdown = File.ReadAllText(file);
    foreach (Match match in Patterns.Code().Matches(markdown))
    {
        string code = Patterns.Using().Replace(match.Groups[1].Value, "");
        if (!compiled.Contains(code, StringComparison.Ordinal))
            throw new InvalidOperationException($"Uncompiled or stale example in {file}.");
        snippets++;
    }
    foreach (Match match in Patterns.Link().Matches(markdown))
    {
        string target = match.Groups[1].Value;
        if (Uri.TryCreate(target, UriKind.Absolute, out _)) continue;
        string[] parts = target.Split('#', 2);
        string path = parts[0].Length == 0 ? file : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file)!, Uri.UnescapeDataString(parts[0])));
        if (!File.Exists(path) && !Directory.Exists(path))
            throw new InvalidOperationException($"Broken documentation link in {file}: {target}");
        if (parts.Length == 2 && path.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
        {
            string fragment = Uri.UnescapeDataString(parts[1]);
            bool found = Patterns.Heading().Matches(File.ReadAllText(path)).Any(heading => Slug(heading.Groups[1].Value) == fragment);
            if (!found) throw new InvalidOperationException($"Missing heading in {file}: {target}");
        }
        links++;
    }
}

// Exercise document-only behavior without fetching models or processing a voice.
Examples.Example16();
Console.WriteLine($"Documentation checks passed: {snippets} compiled snippets, {links} local links; JSON round-trip passed.");

static string Slug(string heading) => Patterns.Punctuation().Replace(heading.ToLowerInvariant(), "").Replace(' ', '-');

internal static partial class Patterns
{
    [GeneratedRegex("^```csharp\\r?\\n(.*?)^```", RegexOptions.Multiline | RegexOptions.Singleline)]
    internal static partial Regex Code();
    [GeneratedRegex("^using [\\w.]+;\\r?\\n", RegexOptions.Multiline)]
    internal static partial Regex Using();
    [GeneratedRegex(@"\]\(([^\s)]+)\)")]
    internal static partial Regex Link();
    [GeneratedRegex(@"^#{1,6} (.+?)\r?$", RegexOptions.Multiline)]
    internal static partial Regex Heading();
    [GeneratedRegex(@"[^\p{L}\p{N} _-]")]
    internal static partial Regex Punctuation();
}
