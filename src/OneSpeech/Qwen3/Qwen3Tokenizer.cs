using System.Globalization;
using System.Text;
using System.Text.Json;

namespace OneSpeech.Qwen3;

/// <summary>Qwen3-ASR byte-level BPE from local model data. Text is normalized to Unicode NFC; no task tokens are added.</summary>
public sealed class Qwen3Tokenizer
{
    private readonly Dictionary<string, int> vocabulary = new(StringComparer.Ordinal);
    private readonly Dictionary<(string Left, string Right), int> ranks = [];
    private readonly Dictionary<string, int> added = new(StringComparer.Ordinal);
    private readonly string[] pieces;
    private readonly byte[][] bytes;
    private readonly char[] byteCharacters = new char[256];
    private readonly Dictionary<char, byte> characterBytes = [];
    private readonly HashSet<int> specialIds = [];
    private readonly HashSet<int> omittedIds = [];
    private readonly HashSet<int> addedIds = [];

    /// <summary>Loads vocab.json, merges.txt and tokenizer_config.json from a local directory. Vocabulary is limited to 200,000 entries, each file to 16 MiB.</summary>
    /// <exception cref="IOException">A data file cannot be read.</exception>
    /// <exception cref="InvalidDataException">Vocabulary or merges exceed the supported bounds or are inconsistent.</exception>
    /// <exception cref="JsonException">JSON data is malformed.</exception>
    public Qwen3Tokenizer(string directory)
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
            foreach (var entry in doc.RootElement.EnumerateObject()) Add(entry.Name, entry.Value.GetInt32(), entry.Name == "<|endoftext|>");
        using (var doc = LoadJson(Path.Combine(directory, "tokenizer_config.json")))
        {
            if (doc.RootElement.GetProperty("tokenizer_class").GetString() != "Qwen2Tokenizer")
                throw new InvalidDataException("Expected Qwen2Tokenizer model data.");
            foreach (var entry in doc.RootElement.GetProperty("added_tokens_decoder").EnumerateObject())
            {
                var value = entry.Value;
                if (!int.TryParse(entry.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int id) ||
                    value.GetProperty("lstrip").GetBoolean() ||
                    value.GetProperty("rstrip").GetBoolean() || value.GetProperty("single_word").GetBoolean() ||
                    value.GetProperty("normalized").GetBoolean())
                    throw new InvalidDataException("Unsupported added-token behavior.");
                Add(value.GetProperty("content").GetString() ?? throw new InvalidDataException("Missing token content."), id, value.GetProperty("special").GetBoolean(), true);
            }
            omittedIds.UnionWith(specialIds);
            if (doc.RootElement.TryGetProperty("additional_special_tokens", out var named))
            {
                // Observed in the toolkit's pinned Transformers 4.57.6: omission uses the named
                // special-token list, not every added-token special flag (blank tokens survive).
                omittedIds.Clear();
                foreach (var token in named.EnumerateArray()) Omit(token);
                foreach (string name in new[] { "bos_token", "eos_token", "pad_token", "unk_token" })
                    if (doc.RootElement.TryGetProperty(name, out var token) && token.ValueKind != JsonValueKind.Null) Omit(token);
            }
            void Omit(JsonElement token)
            {
                string text = token.GetString() ?? throw new InvalidDataException("Missing named special token.");
                if (!vocabulary.TryGetValue(text, out int id)) throw new InvalidDataException("Named special token is absent from vocabulary.");
                omittedIds.Add(id);
            }
        }
        if (vocabulary.Count == 0) throw new InvalidDataException("Empty vocabulary.");
        int count = vocabulary.Values.Max() + 1;
        pieces = new string[count]; bytes = new byte[count][];
        foreach (var pair in vocabulary)
        {
            if (pieces[pair.Value] != null) throw new InvalidDataException("Duplicate vocabulary ID.");
            pieces[pair.Value] = pair.Key;
            if (!addedIds.Contains(pair.Value))
            {
                byte[] decoded = new byte[pair.Key.Length];
                for (int i = 0; i < decoded.Length; i++)
                    if (!characterBytes.TryGetValue(pair.Key[i], out decoded[i])) throw new InvalidDataException("Vocabulary is not byte-level BPE.");
                bytes[pair.Value] = decoded;
            }
            else bytes[pair.Value] = Encoding.UTF8.GetBytes(pair.Key);
        }
        foreach (char character in byteCharacters)
            if (!vocabulary.ContainsKey(character.ToString())) throw new InvalidDataException("Vocabulary lacks a byte token.");
        string mergePath = Path.Combine(directory, "merges.txt");
        if (new FileInfo(mergePath).Length > 16 * 1024 * 1024) throw new InvalidDataException("Merge file too large.");
        int rank = 0;
        foreach (string line in File.ReadLines(mergePath, new UTF8Encoding(false, true)))
        {
            if (line.StartsWith("#version:", StringComparison.Ordinal) || line.Length == 0) continue;
            string[] pair = line.Split(' ');
            if (pair.Length != 2 || rank >= 200000 || !vocabulary.ContainsKey(pair[0]) || !vocabulary.ContainsKey(pair[1]) || !vocabulary.ContainsKey(pair[0] + pair[1]) || !ranks.TryAdd((pair[0], pair[1]), rank++)) throw new InvalidDataException("Invalid BPE merge list.");
        }
    }

    private static JsonDocument LoadJson(string path)
    {
        using var file = File.OpenRead(path);
        if (file.Length > 16 * 1024 * 1024) throw new InvalidDataException("Tokenizer data exceeds 16 MiB.");
        using var reader = new StreamReader(file, new UTF8Encoding(false, true), detectEncodingFromByteOrderMarks: false);
        try { return JsonDocument.Parse(reader.ReadToEnd(), new JsonDocumentOptions { MaxDepth = 8 }); }
        catch (DecoderFallbackException ex) { throw new InvalidDataException("Tokenizer JSON is not UTF-8.", ex); }
    }
    private void Add(string token, int id, bool isSpecial, bool isAdded = false)
    {
        if (id is < 0 or >= 200000 || token.Length is 0 or > 65536 || vocabulary.Count >= 200000) throw new InvalidDataException("Vocabulary entry exceeds bounds.");
        if (vocabulary.TryGetValue(token, out int existing))
        {
            if (existing != id) throw new InvalidDataException("Inconsistent added token.");
        }
        else vocabulary.Add(token, id);
        if (isSpecial) specialIds.Add(id);
        if (isAdded || isSpecial) { added[token] = id; addedIds.Add(id); }
    }

    /// <summary>Returns a data-defined token ID, including a special or timestamp token. Throws KeyNotFoundException if absent.</summary>
    public int GetToken(string token) => vocabulary[token];
    internal bool TryGetToken(string token, out int id) => vocabulary.TryGetValue(token, out id);

    /// <summary>Returns true only for a token marked special by model data. The aligner timestamp slot token is not special.</summary>
    public bool IsSpecial(int id) => specialIds.Contains(id);

    /// <summary>Encodes text with Unicode NFC normalization. Special strings are refused by default unless explicitly allowed. Input is limited to 1 MiB UTF-8.</summary>
    /// <exception cref="ArgumentException">Input contains invalid UTF-16, an unallowed special token or exceeds bounds.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested between pieces or merges.</exception>
    public int[] Encode(string text, bool allowSpecialTokens = false, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        try { if (new UTF8Encoding(false, true).GetByteCount(text) > 1024 * 1024) throw new ArgumentException("Text exceeds 1 MiB UTF-8.", nameof(text)); }
        catch (EncoderFallbackException ex) { throw new ArgumentException("Invalid UTF-16.", nameof(text), ex); }
        cancellationToken.ThrowIfCancellationRequested();
        // Observed in the released Transformers 5.22.2 Qwen2Tokenizer public API.
        text = text.Normalize(NormalizationForm.FormC);
        List<int> result = [];
        int cursor = 0;
        while (cursor < text.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int next = text.Length;
            string? nextSpecial = null;
            foreach (string token in added.Keys)
            {
                int found = text.IndexOf(token, cursor, StringComparison.Ordinal);
                if (found >= 0 && (found < next || found == next && token.Length > (nextSpecial?.Length ?? 0))) { next = found; nextSpecial = token; }
            }
            if (nextSpecial != null && specialIds.Contains(added[nextSpecial]) && !allowSpecialTokens) throw new ArgumentException("Text contains a special token; explicitly allow special tokens if intended.", nameof(text));
            EncodeOrdinary(text[cursor..next], result, cancellationToken);
            if (nextSpecial == null) break;
            result.Add(added[nextSpecial]); cursor = next + nextSpecial.Length;
        }
        return result.ToArray();
    }
    private void EncodeOrdinary(string text, List<int> result, CancellationToken cancellationToken)
    {
        foreach (string piece in Split(text))
        {
            cancellationToken.ThrowIfCancellationRequested();
            byte[] encoded = Encoding.UTF8.GetBytes(piece);
            string[] symbols = new string[encoded.Length];
            int[] previous = new int[encoded.Length], next = new int[encoded.Length], versions = new int[encoded.Length];
            for (int i = 0; i < encoded.Length; i++)
            {
                symbols[i] = byteCharacters[encoded[i]].ToString();
                previous[i] = i - 1;
                next[i] = i + 1 < encoded.Length ? i + 1 : -1;
            }
            // At most three candidates are queued per merge; versions invalidate stale pairs.
            // Rank ties go left-to-right, independently of dictionary enumeration order.
            var queue = new PriorityQueue<(int Left, int Right, int LeftVersion, int RightVersion), (int Rank, int Position)>();
            void Enqueue(int left)
            {
                if (left < 0 || next[left] < 0) return;
                int right = next[left];
                if (ranks.TryGetValue((symbols[left], symbols[right]), out int rank))
                    queue.Enqueue((left, right, versions[left], versions[right]), (rank, left));
            }
            for (int i = 0; i < encoded.Length - 1; i++) Enqueue(i);
            while (queue.TryDequeue(out var pair, out _))
            {
                cancellationToken.ThrowIfCancellationRequested();
                int left = pair.Left, right = pair.Right;
                if (versions[left] != pair.LeftVersion || versions[right] != pair.RightVersion || next[left] != right) continue;
                symbols[left] += symbols[right];
                versions[left]++; versions[right]++;
                next[left] = next[right];
                if (next[right] >= 0) previous[next[right]] = left;
                Enqueue(previous[left]); Enqueue(left);
            }
            for (int i = 0; i >= 0 && i < encoded.Length; i = next[i]) result.Add(vocabulary[symbols[i]]);
        }
    }
    /// <summary>Decodes IDs as UTF-8, replacing incomplete byte sequences. By default the model's named special tokens are omitted, matching the toolkit's Transformers 4.57.6; blank and timestamp tokens are retained. If no named list exists, added-token special flags determine omission.</summary>
    /// <exception cref="ArgumentOutOfRangeException">An ID is not in the vocabulary.</exception>
    public string Decode(ReadOnlySpan<int> tokens, bool includeSpecialTokens = false)
    {
        using var output = new MemoryStream();
        foreach (int token in tokens)
        {
            if ((uint)token >= bytes.Length || bytes[token] == null) throw new ArgumentOutOfRangeException(nameof(tokens));
            if (includeSpecialTokens || !omittedIds.Contains(token)) output.Write(bytes[token]);
        }
        return Encoding.UTF8.GetString(output.GetBuffer(), 0, checked((int)output.Length));
    }

    // Scalar-aware splitting: contractions, optional-prefix letters, individual numbers,
    // punctuation with newline suffixes, then whitespace. Branch order is significant.
    private static IEnumerable<string> Split(string text)
    {
        int cursor = 0;
        while (cursor < text.Length)
        {
            int begin = cursor;
            string? contraction = null;
            foreach (string suffix in new[] { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" })
                if (text.AsSpan(cursor).StartsWith(suffix, StringComparison.OrdinalIgnoreCase)) { contraction = suffix; break; }
            if (contraction != null) { cursor += contraction.Length; yield return text[begin..cursor]; continue; }
            Rune first = Rune.GetRuneAt(text, cursor);
            int next = cursor + first.Utf16SequenceLength;
            int letters = Class(first) == 1 ? cursor : -1;
            if (letters < 0 && first.Value is not ('\r' or '\n') && Class(first) != 2 &&
                next < text.Length && Class(Rune.GetRuneAt(text, next)) == 1) letters = next;
            if (letters >= 0)
            {
                cursor = letters;
                while (cursor < text.Length && Class(Rune.GetRuneAt(text, cursor)) == 1)
                    cursor += Rune.GetRuneAt(text, cursor).Utf16SequenceLength;
            }
            else if (Class(first) == 2) cursor = next;
            else
            {
                int punctuation = first.Value == ' ' && next < text.Length ? next : begin;
                if (Class(Rune.GetRuneAt(text, punctuation)) == 3)
                {
                    cursor = punctuation;
                    while (cursor < text.Length && Class(Rune.GetRuneAt(text, cursor)) == 3)
                        cursor += Rune.GetRuneAt(text, cursor).Utf16SequenceLength;
                    while (cursor < text.Length && text[cursor] is '\r' or '\n') cursor++;
                }
                else
                {
                    int last = cursor, newlineEnd = -1;
                    while (cursor < text.Length && Rune.IsWhiteSpace(Rune.GetRuneAt(text, cursor)))
                    {
                        last = cursor;
                        if (text[cursor] is '\r' or '\n') newlineEnd = cursor + 1;
                        cursor += Rune.GetRuneAt(text, cursor).Utf16SequenceLength;
                    }
                    if (newlineEnd >= 0) cursor = newlineEnd;
                    else if (cursor < text.Length && last > begin) cursor = last;
                }
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
