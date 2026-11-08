using System.Globalization;
using System.Text;
using System.Text.Json;

namespace OneSpeech.Whisper;

/// <summary>Whisper byte-level BPE from local vocabulary and merge data. No normalization or implicit task tokens are added.</summary>
public sealed class WhisperTokenizer
{
    private readonly Dictionary<string, int> vocabulary = new(StringComparer.Ordinal);
    private readonly Dictionary<(string Left, string Right), int> ranks = [];
    private readonly Dictionary<string, int> special = new(StringComparer.Ordinal);
    private readonly string[] pieces;
    private readonly byte[][] bytes;
    private readonly char[] byteCharacters = new char[256];
    private readonly Dictionary<char, byte> characterBytes = [];
    private readonly HashSet<int> specialIds = [];

    /// <summary>Loads vocab.json, merges.txt and added_tokens.json from a local directory. Vocabulary is limited to 60,000 entries, each file to 8 MiB.</summary>
    /// <exception cref="IOException">A data file cannot be read.</exception>
    /// <exception cref="InvalidDataException">Vocabulary or merges exceed the supported bounds or are inconsistent.</exception>
    /// <exception cref="JsonException">JSON data is malformed.</exception>
    public WhisperTokenizer(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        int extra = 256;
        for (int b = 0; b < 256; b++)
        {
            char character = b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? (char)b : (char)extra++;
            byteCharacters[b] = character;
            characterBytes.Add(character, (byte)b);
        }
        using (var doc = LoadJson(Path.Combine(directory, "vocab.json")))
            foreach (var entry in doc.RootElement.EnumerateObject()) Add(entry.Name, entry.Value.GetInt32(), false);
        using (var doc = LoadJson(Path.Combine(directory, "added_tokens.json")))
            foreach (var entry in doc.RootElement.EnumerateObject()) Add(entry.Name, entry.Value.GetInt32(), true);
        int count = vocabulary.Values.Max() + 1;
        pieces = new string[count]; bytes = new byte[count][];
        foreach (var pair in vocabulary)
        {
            if (pieces[pair.Value] != null) throw new InvalidDataException("Duplicate vocabulary ID.");
            pieces[pair.Value] = pair.Key;
            if (!specialIds.Contains(pair.Value))
            {
                byte[] decoded = new byte[pair.Key.Length];
                for (int i = 0; i < decoded.Length; i++)
                    if (!characterBytes.TryGetValue(pair.Key[i], out decoded[i])) throw new InvalidDataException("Vocabulary is not byte-level BPE.");
                bytes[pair.Value] = decoded;
            }
            else bytes[pair.Value] = Encoding.UTF8.GetBytes(pair.Key);
        }
        string mergePath = Path.Combine(directory, "merges.txt");
        if (new FileInfo(mergePath).Length > 8 * 1024 * 1024) throw new InvalidDataException("Merge file too large.");
        int rank = 0;
        foreach (string line in File.ReadLines(mergePath, new UTF8Encoding(false, true)))
        {
            if (line.StartsWith("#version:", StringComparison.Ordinal) || line.Length == 0) continue;
            string[] pair = line.Split(' ');
            if (pair.Length != 2 || rank >= 60000 || !ranks.TryAdd((pair[0], pair[1]), rank++)) throw new InvalidDataException("Invalid BPE merge list.");
        }
    }

    private static JsonDocument LoadJson(string path)
    {
        if (new FileInfo(path).Length > 8 * 1024 * 1024) throw new InvalidDataException("Tokenizer data exceeds 8 MiB.");
        using var file = File.OpenRead(path);
        return JsonDocument.Parse(file, new JsonDocumentOptions { MaxDepth = 8 });
    }
    private void Add(string token, int id, bool isSpecial)
    {
        if (id is < 0 or >= 60000 || token.Length > 65536 || vocabulary.Count >= 60000) throw new InvalidDataException("Vocabulary entry exceeds bounds.");
        if (vocabulary.TryGetValue(token, out int existing))
        {
            if (existing != id) throw new InvalidDataException("Inconsistent added token.");
        }
        else vocabulary.Add(token, id);
        if (isSpecial) { special.Add(token, id); specialIds.Add(id); }
    }

    /// <summary>Returns a data-defined token ID, including a special or timestamp token. Throws KeyNotFoundException if absent.</summary>
    public int GetToken(string token) => vocabulary[token];

    /// <summary>Returns true for a data-defined special token, including timestamp tokens.</summary>
    public bool IsSpecial(int id) => specialIds.Contains(id);

    /// <summary>Encodes UTF-8 text exactly, without normalization. Special strings are refused unless explicitly allowed. Input is limited to 1 MiB UTF-8.</summary>
    /// <exception cref="ArgumentException">Input contains invalid UTF-16, an unallowed special token or exceeds bounds.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested between pieces or merges.</exception>
    public int[] Encode(string text, bool allowSpecialTokens = false, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        try { if (new UTF8Encoding(false, true).GetByteCount(text) > 1024 * 1024) throw new ArgumentException("Text exceeds 1 MiB UTF-8.", nameof(text)); }
        catch (EncoderFallbackException ex) { throw new ArgumentException("Invalid UTF-16.", nameof(text), ex); }
        List<int> result = [];
        int cursor = 0;
        while (cursor < text.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int next = text.Length;
            string? nextSpecial = null;
            foreach (string token in special.Keys)
            {
                int found = text.IndexOf(token, cursor, StringComparison.Ordinal);
                if (found >= 0 && (found < next || found == next && token.Length > (nextSpecial?.Length ?? 0))) { next = found; nextSpecial = token; }
            }
            if (nextSpecial != null && !allowSpecialTokens) throw new ArgumentException("Text contains a special token; explicitly allow special tokens if intended.", nameof(text));
            EncodeOrdinary(text[cursor..next], result, cancellationToken);
            if (nextSpecial == null) break;
            result.Add(special[nextSpecial]); cursor = next + nextSpecial.Length;
        }
        return result.ToArray();
    }
    private void EncodeOrdinary(string text, List<int> result, CancellationToken cancellationToken)
    {
        foreach (string piece in Split(text))
        {
            cancellationToken.ThrowIfCancellationRequested();
            List<string> symbols = Encoding.UTF8.GetBytes(piece).Select(b => byteCharacters[b].ToString()).ToList();
            while (symbols.Count > 1)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int best = int.MaxValue, position = -1;
                for (int i = 0; i < symbols.Count - 1; i++)
                    if (ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < best) { best = rank; position = i; }
                if (position < 0) break;
                string left = symbols[position], right = symbols[position + 1];
                for (int i = 0; i < symbols.Count - 1; i++)
                    if (symbols[i] == left && symbols[i + 1] == right) { symbols[i] += symbols[i + 1]; symbols.RemoveAt(i + 1); }
            }
            foreach (string symbol in symbols) result.Add(vocabulary[symbol]);
        }
    }

    /// <summary>Decodes IDs as UTF-8, replacing incomplete byte sequences. By default special and timestamp tokens are omitted.</summary>
    /// <exception cref="ArgumentOutOfRangeException">An ID is not in the vocabulary.</exception>
    public string Decode(ReadOnlySpan<int> tokens, bool includeSpecialTokens = false)
    {
        using var output = new MemoryStream();
        foreach (int token in tokens)
        {
            if ((uint)token >= bytes.Length || bytes[token] == null) throw new ArgumentOutOfRangeException(nameof(tokens));
            if (includeSpecialTokens || !specialIds.Contains(token)) output.Write(bytes[token]);
        }
        return Encoding.UTF8.GetString(output.GetBuffer(), 0, checked((int)output.Length));
    }

    // Scalar-aware equivalent of GPT-2's Unicode pre-tokenization pattern. Astral letters
    // must remain letters rather than being treated as two UTF-16 punctuation characters.
    private static IEnumerable<string> Split(string text)
    {
        int cursor = 0;
        while (cursor < text.Length)
        {
            int begin = cursor;
            string? contraction = null;
            foreach (string suffix in new[] { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" })
                if (text.AsSpan(cursor).StartsWith(suffix, StringComparison.Ordinal)) { contraction = suffix; break; }
            if (contraction != null) { cursor += contraction.Length; yield return text[begin..cursor]; continue; }
            int start = text[cursor] == ' ' && cursor + 1 < text.Length ? cursor + 1 : cursor;
            Rune first = Rune.GetRuneAt(text, start);
            int category = Class(first);
            if (category != 0)
            {
                cursor = start + first.Utf16SequenceLength;
                while (cursor < text.Length && Class(Rune.GetRuneAt(text, cursor)) == category) cursor += Rune.GetRuneAt(text, cursor).Utf16SequenceLength;
            }
            else
            {
                cursor = begin;
                int last = cursor;
                while (cursor < text.Length && Rune.IsWhiteSpace(Rune.GetRuneAt(text, cursor))) { last = cursor; cursor += Rune.GetRuneAt(text, cursor).Utf16SequenceLength; }
                if (cursor < text.Length && last > begin) cursor = last;
            }
            if (cursor == begin) throw new InvalidDataException("Tokenizer failed to advance.");
            yield return text[begin..cursor];
        }
    }
    private static int Class(Rune rune)
    {
        if (Rune.IsWhiteSpace(rune)) return 0;
        var category = Rune.GetUnicodeCategory(rune);
        if (category is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter) return 1;
        return category is UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber ? 2 : 3;
    }
}
