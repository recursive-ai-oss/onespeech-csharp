using System.Collections.Immutable;
using System.Text;
using OneSpeech.Decoding;

namespace OneSpeech.Custom;

/// <summary>Immutable ID-to-piece decoding vocabulary. Pieces concatenate as text; optional U+2581 metaspace markers become spaces. No encoding, normalization, byte fallback or model-specific special-token policy is guessed.</summary>
public sealed class TokenVocabulary
{
    internal readonly ImmutableArray<string> Pieces;
    internal readonly bool Metaspace;
    internal readonly bool ExcludeTrailingPunctuation;
    /// <summary>Copies 1..1000000 nonnull pieces of at most 1024 UTF-16 characters each. Include a piece for blank. For model-specific unknown surfaces supply the desired decoded string as that ID's piece. Metaspace defaults false; excluding trailing punctuation from timing defaults false. Invalid pieces throw InvalidModelShapeException.</summary>
    public TokenVocabulary(IEnumerable<string> pieces, bool metaspace = false, bool excludeTrailingPunctuationFromTiming = false)
    {
        ArgumentNullException.ThrowIfNull(pieces);
        var result = ImmutableArray.CreateBuilder<string>();
        foreach (string piece in pieces)
        {
            if (result.Count == 1000000 || piece is null || piece.Length > 1024) throw new InvalidModelShapeException("Vocabulary exceeds its bounds or has a null piece.");
            result.Add(piece);
        }
        if (result.Count == 0) throw new InvalidModelShapeException("Vocabulary must include blank.");
        Pieces = result.ToImmutable(); Metaspace = metaspace; ExcludeTrailingPunctuation = excludeTrailingPunctuationFromTiming;
    }

    internal (string Text, WordTokenRange[] Words, FrameToken[] Timings) Decode(ImmutableArray<FrameToken> tokens)
    {
        var all = new StringBuilder(); var current = new StringBuilder(); var words = new List<WordTokenRange>();
        int first = 0, last = 0;
        var timing = tokens.ToArray();
        void Flush()
        {
            if (current.Length == 0) return;
            int count = last - first + 1;
            if (ExcludeTrailingPunctuation)
                while (count > 1 && Pieces[tokens[first + count - 1].Id].TrimStart('▁') is { Length: > 0 } p && p.EnumerateRunes().All(Rune.IsPunctuation)) count--;
            words.Add(new(current.ToString(), first, count)); current.Clear();
        }
        for (int i = 0; i < tokens.Length; i++)
        {
            if ((uint)tokens[i].Id >= Pieces.Length) throw new InvalidModelShapeException("Decoder returned an ID outside the vocabulary.");
            string piece = Pieces[tokens[i].Id];
            if (Metaspace) piece = piece.Replace('▁', ' ');
            if (ExcludeTrailingPunctuation && piece.Trim() is { Length: > 0 } punctuation && punctuation.EnumerateRunes().All(Rune.IsPunctuation))
                timing[i] = timing[i] with { EndFrame = timing[i].StartFrame };
            all.Append(piece);
            foreach (char ch in piece)
            {
                if (char.IsWhiteSpace(ch)) Flush();
                else { if (current.Length == 0) first = i; last = i; current.Append(ch); }
            }
        }
        Flush();
        string text = all.ToString();
        if (Metaspace && text.StartsWith(' ')) text = text[1..];
        return (text, words.ToArray(), timing);
    }
}
