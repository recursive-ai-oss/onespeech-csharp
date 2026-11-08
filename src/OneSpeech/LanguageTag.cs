using System.Globalization;
using OneSpeech.Catalog;

namespace OneSpeech;

// Models consume language-only tags. Resolve names/ISO aliases once, independent of caller culture.
internal static class LanguageTag
{
    private static readonly Dictionary<string, string> Aliases = Build();
    private static Dictionary<string, string> Build()
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string tag in ModelCatalog.Default.Entries.SelectMany(e => e.Capabilities.Languages).Distinct())
        {
            string primary = tag.Split('-', '_')[0].ToLowerInvariant();
            result.TryAdd(primary, primary);
            try
            {
                var culture = CultureInfo.GetCultureInfo(primary);
                result.TryAdd(culture.EnglishName, primary);
                result.TryAdd(culture.ThreeLetterISOLanguageName, primary);
            }
            catch (CultureNotFoundException) { /* Model-specific tags remain usable as declared. */ }
        }
        result["English"] = "en"; result["Chinese"] = "zh"; result["Cantonese"] = "yue";
        result["eng"] = "en"; result["zho"] = "zh"; result["chi"] = "zh";
        return result;
    }

    internal static string? Normalize(string? language)
    {
        if (language is null) return null;
        string value = language.Trim();
        if (Aliases.TryGetValue(value, out string? tag)) return tag;
        string[] parts = value.Replace('_', '-').Split('-');
        if (parts.Length > 1 && parts.All(p => p.Length is > 0 and <= 8 && p.All(char.IsAsciiLetterOrDigit)) && Aliases.TryGetValue(parts[0], out tag)) return tag;
        throw new CapabilityNotSupportedException($"Unsupported language '{language}'. Use a declared language tag such as 'en' or omit the hint.");
    }

    internal static string? Require(string? language, Capabilities capabilities, string subject)
    {
        string? tag = Normalize(language);
        if (tag is not null && !capabilities.Languages.Any(l => Normalize(l) == tag))
            throw new CapabilityNotSupportedException($"Model '{subject}' does not support language '{language}' (normalized to '{tag}').");
        return tag;
    }
}
