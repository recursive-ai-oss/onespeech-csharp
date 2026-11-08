using System.Collections.Immutable;

namespace OneSpeech.Qwen3;

internal static class Qwen3Languages
{
    internal static readonly ImmutableArray<string> Tags = ["zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "ro", "hu", "mk"];
    private static readonly string[] Names = ["Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"];
    internal static string Name(string tag)
    {
        int index = Tags.IndexOf(tag);
        return index < 0 ? throw new CapabilityNotSupportedException("Qwen3-ASR does not declare language '" + tag + "'.") : Names[index];
    }
    internal static string? Tag(string name)
    {
        int index = Array.IndexOf(Names, name);
        return index < 0 ? null : Tags[index];
    }
}
