using System.Collections.Immutable;
using System.Text;

namespace OneSpeech;

/// <summary>
/// Word and segment spacing shared by every family: words carry no leading or trailing whitespace, and a
/// separating space is added only between scripts that are written with spaces. Whisper's tokenizer output
/// carries the space as part of the next token; Moonshine, Qwen3 and the transducers do not.
/// </summary>
internal static class TextSpacing
{
    /// <summary>Trims a segment's text and its words' text; words that are only whitespace are dropped. Returns the same instance when nothing changes.</summary>
    internal static Segment Clean(Segment segment)
    {
        string text = segment.Text.Trim();
        bool dirty = text.Length != segment.Text.Length;
        foreach (var word in segment.Words)
            if (word.Text.Length == 0 || char.IsWhiteSpace(word.Text[0]) || char.IsWhiteSpace(word.Text[^1])) { dirty = true; break; }
        if (!dirty) return segment;
        var words = ImmutableArray.CreateBuilder<Word>(segment.Words.Length);
        foreach (var word in segment.Words)
        {
            string trimmed = word.Text.Trim();
            if (trimmed.Length != 0) words.Add(trimmed.Length == word.Text.Length ? word : word with { Text = trimmed });
        }
        return segment with { Text = text, Words = words.ToImmutable() };
    }

    /// <summary>True when a space belongs between text ending in <paramref name="before"/> and text starting with <paramref name="after"/>.</summary>
    internal static bool NeedsSpace(string before, string after)
    {
        if (before.Length == 0 || after.Length == 0 || char.IsWhiteSpace(before[^1]) || char.IsWhiteSpace(after[0])) return false;
        Rune.TryGetRuneAt(before, before.Length - 1 - (char.IsLowSurrogate(before[^1]) && before.Length > 1 ? 1 : 0), out Rune last);
        Rune.TryGetRuneAt(after, 0, out Rune first);
        return !NoSpaceScript(last) && !NoSpaceScript(first);
    }

    /// <summary>Joins pieces with one space where the script calls for it; empty pieces are skipped and edges trimmed.</summary>
    internal static string Join(IEnumerable<string> pieces)
    {
        var builder = new StringBuilder();
        foreach (string raw in pieces)
        {
            string piece = raw.Trim();
            if (piece.Length == 0) continue;
            if (NeedsSpace(builder.Length == 0 ? "" : builder.ToString(Math.Max(0, builder.Length - 2), Math.Min(2, builder.Length)), piece)) builder.Append(' ');
            builder.Append(piece);
        }
        return builder.ToString();
    }

    // Scripts written without spaces between words: Han, kana, CJK and fullwidth punctuation, Thai, Lao, Myanmar, Khmer (Unicode blocks).
    private static bool NoSpaceScript(Rune r)
    {
        int v = r.Value;
        return v is >= 0x3000 and <= 0x30FF or >= 0x3400 and <= 0x4DBF or >= 0x4E00 and <= 0x9FFF or >= 0xF900 and <= 0xFAFF
            or >= 0xFF00 and <= 0xFFEF or >= 0x0E00 and <= 0x0EFF or >= 0x1000 and <= 0x109F or >= 0x1780 and <= 0x17FF or >= 0x20000 and <= 0x3FFFF;
    }
}
