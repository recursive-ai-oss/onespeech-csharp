using System.Text;
using System.Text.Json;
using OneSpeech.Decoding;

namespace OneSpeech.FastConformer;

// Contextual phrases additionally use the publisher's BPE merge ranks and normalization data.
// The pinned tokenizer declares Metaspace decoding (no ByteFallback decoder).
internal sealed class ParakeetVocabulary
{
    private readonly string[] pieces = new string[8192];
    private ParakeetPhraseTokenizer? phraseTokenizer;
    internal System.Collections.Immutable.ImmutableArray<int> EncodePhrase(string text, CancellationToken ct)
    {
        var ids = (phraseTokenizer ?? throw new CapabilityNotSupportedException("Tokenizer data does not include phrase encoding.")).Encode(text, ct);
        if (ids.Contains(0)) throw new CapabilityNotSupportedException("A contextual phrase contains characters unknown to this tokenizer; biasing the shared unknown ID is refused.");
        return ids;
    }
    internal ParakeetVocabulary(string path) : this(ReadFile(path)) { }

    private static byte[] ReadFile(string path)
    {
        if (new FileInfo(path).Length > 2 * 1024 * 1024) throw new InvalidDataException("Parakeet tokenizer exceeds 2 MiB.");
        return File.ReadAllBytes(path);
    }

    internal ParakeetVocabulary(ReadOnlyMemory<byte> bytes)
    {
        if (bytes.Length > 2 * 1024 * 1024) throw new InvalidDataException("Parakeet tokenizer exceeds 2 MiB.");
        try { Parse(bytes); }
        catch (Exception exception) when (exception is JsonException or KeyNotFoundException or InvalidOperationException or FormatException or ArgumentException or IndexOutOfRangeException)
        { throw new InvalidDataException("Invalid Parakeet vocabulary.", exception); }
    }

    private void Parse(ReadOnlyMemory<byte> bytes)
    {
        _ = new UTF8Encoding(false, true).GetCharCount(bytes.Span);
        using var document = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 16 });
        var model = document.RootElement.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE") throw new InvalidDataException("Unsupported Parakeet vocabulary.");
        int count = 0;
        HashSet<string> names = new(StringComparer.Ordinal);
        foreach (var property in model.GetProperty("vocab").EnumerateObject())
        {
            int id = property.Value.GetInt32();
            if ((uint)id >= pieces.Length || pieces[id] != null || !names.Add(property.Name) || Encoding.UTF8.GetByteCount(property.Name) > 1024) throw new InvalidDataException("Invalid Parakeet vocabulary ID or piece.");
            pieces[id] = property.Name; count++;
        }
        if (count != pieces.Length) throw new InvalidDataException("Incomplete Parakeet vocabulary.");
        if (document.RootElement.TryGetProperty("normalizer", out _) && model.TryGetProperty("merges", out _))
            phraseTokenizer = new ParakeetPhraseTokenizer(document.RootElement, pieces);
    }

    internal string Decode(ReadOnlySpan<int> ids)
    {
        var text = new StringBuilder();
        foreach (int id in ids)
        {
            if ((uint)id >= pieces.Length) throw new InvalidDataException("Invalid Parakeet token ID.");
            // SentencePiece's unknown surface is spaced U+2047, as observed through NeMo's public tokenizer.
            text.Append(pieces[id] == "<unk>" ? " ⁇ " : pieces[id]);
        }
        string result = text.ToString();
        if (result.StartsWith('▁')) result = result[1..];
        return result.Replace('▁', ' ');
    }

    internal FrameToken[] TimingTokens(ReadOnlySpan<FrameToken> tokens)
    {
        FrameToken[] result = tokens.ToArray();
        for (int i = 0; i < result.Length; i++)
        {
            int id = result[i].Id;
            if ((uint)id >= pieces.Length) throw new InvalidDataException("Invalid Parakeet token ID.");
            string piece = pieces[id].TrimStart('▁');
            // NeMo timestamp output assigns punctuation-only emissions zero duration,
            // even when the TDT duration head advances the acoustic clock on that emission.
            if (piece.Length > 0 && piece.EnumerateRunes().All(Rune.IsPunctuation))
                result[i] = result[i] with { EndFrame = result[i].StartFrame };
        }
        return result;
    }

    internal WordTokenRange[] TimingRanges(ReadOnlySpan<FrameToken> tokens, ReadOnlySpan<WordTokenRange> words)
    {
        WordTokenRange[] result = words.ToArray();
        for (int i = 0; i < result.Length; i++)
        {
            var word = result[i];
            int count = word.TokenCount;
            // Native word endpoints stop at the last lexical emission, not a later punctuation emission.
            while (count > 1)
            {
                string piece = pieces[tokens[word.FirstToken + count - 1].Id].TrimStart('▁');
                if (piece.Length == 0 || !piece.EnumerateRunes().All(Rune.IsPunctuation)) break;
                count--;
            }
            result[i] = word with { TokenCount = count };
        }
        return result;
    }

    internal WordTokenRange[] Words(ReadOnlySpan<FrameToken> tokens)
    {
        List<WordTokenRange> words = [];
        var text = new StringBuilder();
        int first = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            int id = tokens[i].Id;
            if ((uint)id >= pieces.Length) throw new InvalidDataException("Invalid Parakeet token ID.");
            string piece = pieces[id] == "<unk>" ? "▁⁇" : pieces[id];
            if (piece.StartsWith('▁') && text.Length > 0)
            {
                words.Add(new(text.ToString(), first, i - first)); text.Clear(); first = i;
            }
            text.Append(piece.Replace('▁', ' '));
            if (text.Length > 0 && text[0] == ' ') text.Remove(0, 1);
        }
        if (text.Length > 0) words.Add(new(text.ToString(), first, tokens.Length - first));
        return words.ToArray();
    }
}
