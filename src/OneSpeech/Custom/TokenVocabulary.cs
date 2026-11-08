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
    private readonly Func<string, CancellationToken, ImmutableArray<int>>? phraseEncoder;
    internal bool CanEncodePhrases => phraseEncoder != null;

    /// <summary>Returns a copy with an explicit phrase encoder for this exact vocabulary. No tokenizer is guessed.
    /// The callback must return 1..256 nonblank ordinary IDs, honor cancellation, and preserve word-boundary markers.
    /// It is invoked when configuring phrases, not per audio frame. The caller owns callback lifetime and thread safety.
    /// Invalid output is rejected by WithPhrases. The existing vocabulary remains unchanged.</summary>
    /// <exception cref="ArgumentNullException">Encoder is null.</exception>
    public TokenVocabulary WithPhraseEncoder(Func<string, CancellationToken, ImmutableArray<int>> encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        return new TokenVocabulary(this, encoder);
    }

    private TokenVocabulary(TokenVocabulary source, Func<string, CancellationToken, ImmutableArray<int>> encoder)
    {
        Pieces = source.Pieces; Metaspace = source.Metaspace; ExcludeTrailingPunctuation = source.ExcludeTrailingPunctuation;
        phraseEncoder = encoder;
    }

    internal ImmutableArray<int> EncodePhrase(string text, CancellationToken ct) =>
        (phraseEncoder ?? throw new CapabilityNotSupportedException("Supply TokenVocabulary.WithPhraseEncoder for contextual biasing."))(text, ct);
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
