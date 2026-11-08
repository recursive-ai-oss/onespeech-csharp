using System.Globalization;
using System.Text;
using System.Text.Json;

namespace OneSpeech.Moonshine;

/// <summary>Moonshine's published SentencePiece-style BPE tokenizer data. Adds BOS on encoding by default, never EOS. No Unicode normalization beyond the data-defined space marker.</summary>
public sealed class MoonshineTokenizer
{
    private readonly Dictionary<string, int> vocabulary = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> merges = [];
    private readonly Dictionary<string, int> specials = new(StringComparer.Ordinal);
    private readonly HashSet<int> specialIds = [];
    private readonly string?[] pieces;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    /// <summary>Reads tokenizer.json from a directory, limited to 8 MiB, 32768 vocabulary IDs and 65536 merges. Throws IOException for missing data, JsonException for malformed JSON, or InvalidDataException for inconsistent/unsupported data.</summary>
    public MoonshineTokenizer(string directory) : this(ReadTokenizer(directory)) { }

    private static byte[] ReadTokenizer(string directory)
    {
        using var input = File.OpenRead(Path.Combine(directory, "tokenizer.json"));
        long length = input.Length;
        if (length > 8 * 1024 * 1024) throw new InvalidDataException("Moonshine tokenizer exceeds 8 MiB.");
        var bytes = new byte[checked((int)length)]; input.ReadExactly(bytes); return bytes;
    }
    internal MoonshineTokenizer(byte[] bytes)
    {
        if (bytes.Length > 8 * 1024 * 1024) throw new InvalidDataException("Moonshine tokenizer exceeds 8 MiB.");
        using var json = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 16 });
        pieces = new string[32768];
        try { Parse(json.RootElement); }
        catch (Exception error) when (error is KeyNotFoundException or InvalidOperationException or FormatException or OverflowException or ArgumentException) { throw new InvalidDataException("Invalid Moonshine tokenizer structure.", error); }
    }
    private void Parse(JsonElement root)
    {
        var normalization = root.GetProperty("normalizer");
        var normalizers = normalization.GetProperty("normalizers");
        var decoding = root.GetProperty("decoder");
        var decoders = decoding.GetProperty("decoders");
        if (root.GetProperty("pre_tokenizer").ValueKind != JsonValueKind.Null || normalization.GetProperty("type").GetString() != "Sequence" || normalizers.GetArrayLength() != 2 ||
            normalizers[0].GetProperty("type").GetString() != "Prepend" || normalizers[0].GetProperty("prepend").GetString() != "▁" ||
            normalizers[1].GetProperty("type").GetString() != "Replace" || normalizers[1].GetProperty("pattern").GetProperty("String").GetString() != " " || normalizers[1].GetProperty("content").GetString() != "▁" ||
            decoding.GetProperty("type").GetString() != "Sequence" || decoders.GetArrayLength() != 4 || decoders[0].GetProperty("type").GetString() != "Replace" ||
            decoders[0].GetProperty("pattern").GetProperty("String").GetString() != "▁" || decoders[0].GetProperty("content").GetString() != " " || decoders[1].GetProperty("type").GetString() != "ByteFallback" ||
            decoders[2].GetProperty("type").GetString() != "Fuse" || decoders[3].GetProperty("type").GetString() != "Strip" || decoders[3].GetProperty("content").GetString() != " " || decoders[3].GetProperty("start").GetInt32() != 1 || decoders[3].GetProperty("stop").GetInt32() != 0)
            throw new InvalidDataException("Unsupported Moonshine tokenizer normalization or decoding.");
        var template = root.GetProperty("post_processor");
        var single = template.GetProperty("single");
        if (template.GetProperty("type").GetString() != "TemplateProcessing" || single.GetArrayLength() != 2 || single[0].GetProperty("SpecialToken").GetProperty("id").GetString() != "<s>" || single[1].GetProperty("Sequence").GetProperty("id").GetString() != "A")
            throw new InvalidDataException("Unsupported Moonshine tokenizer template.");
        var bosIds = template.GetProperty("special_tokens").GetProperty("<s>").GetProperty("ids");
        if (bosIds.GetArrayLength() != 1 || bosIds[0].GetInt32() != 1) throw new InvalidDataException("Unsupported BOS template.");
        var model = root.GetProperty("model");
        foreach (string property in new[] { "continuing_subword_prefix", "end_of_word_suffix" })
            if (model.TryGetProperty(property, out var suffix) && suffix.ValueKind != JsonValueKind.Null) throw new InvalidDataException("Unsupported BPE affix.");
        if (model.TryGetProperty("dropout", out var dropout) && dropout.ValueKind != JsonValueKind.Null && dropout.GetDouble() != 0 || model.TryGetProperty("ignore_merges", out var ignore) && ignore.GetBoolean()) throw new InvalidDataException("Unsupported BPE merge options.");
        if (model.GetProperty("type").GetString() != "BPE" || !model.GetProperty("byte_fallback").GetBoolean()) throw new InvalidDataException("Expected byte-fallback BPE.");
        foreach (var pair in model.GetProperty("vocab").EnumerateObject()) Add(pair.Name, pair.Value.GetInt32(), false);
        foreach (var token in root.GetProperty("added_tokens").EnumerateArray())
        {
            if (!token.GetProperty("special").GetBoolean()) throw new InvalidDataException("Only special added tokens are supported.");
            foreach (string property in new[] { "single_word", "lstrip", "rstrip", "normalized" })
                if (token.TryGetProperty(property, out var flag) && flag.GetBoolean()) throw new InvalidDataException("Unsupported added-token matching rule.");
            Add(token.GetProperty("content").GetString() ?? throw new InvalidDataException("Null added token."), token.GetProperty("id").GetInt32(), true);
        }
        for (int b = 0; b < 256; b++) if (!vocabulary.ContainsKey("<0x" + b.ToString("X2", CultureInfo.InvariantCulture) + ">")) throw new InvalidDataException("Missing byte fallback token.");
        if (!vocabulary.ContainsKey("▁") || !specials.TryGetValue("<s>", out int bos) || bos != 1 || !specials.TryGetValue("</s>", out int eos) || eos != 2) throw new InvalidDataException("Missing Moonshine space marker or BOS/EOS tokens.");
        int rank = 0;
        foreach (var entry in model.GetProperty("merges").EnumerateArray())
        {
            string[] pair = (entry.GetString() ?? throw new InvalidDataException("Null merge.")).Split(' ');
            if (pair.Length != 2 || rank >= 65536 || !merges.TryAdd((pair[0], pair[1]), rank++)) throw new InvalidDataException("Invalid BPE merge.");
        }
    }
    private void Add(string piece, int id, bool special)
    {
        if (id is < 0 or >= 32768 || piece.Length > 4096 || special && piece.Length == 0) throw new InvalidDataException("Tokenizer entry outside bounds.");
        if (vocabulary.TryGetValue(piece, out int existing) && existing != id || pieces[id] is { } old && old != piece) throw new InvalidDataException("Conflicting token ID.");
        vocabulary[piece] = id; pieces[id] = piece;
        if (special) { specials[piece] = id; specialIds.Add(id); }
    }
    /// <summary>Encodes up to 16384 UTF-16 code units, preserving special-token IDs. Empty input returns just BOS if requested. Throws ArgumentOutOfRangeException for larger text, ArgumentNullException for null, ArgumentException for invalid UTF-16, and OperationCanceledException when canceled.</summary>
    public int[] Encode(string text, bool addBeginningOfSequence = true, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        if (text.Length > 16384) throw new ArgumentOutOfRangeException(nameof(text));
        _ = StrictUtf8.GetByteCount(text);
        List<int> output = [];
        if (addBeginningOfSequence) output.Add(1);
        int position = 0;
        while (position < text.Length)
        {
            int next = text.Length;
            string? found = null;
            foreach (var special in specials.Keys)
            {
                int index = text.IndexOf(special, position, StringComparison.Ordinal);
                if (index >= 0 && (index < next || index == next && (found is null || special.Length > found.Length))) { next = index; found = special; }
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (next > position) EncodeOrdinary(text[position..next], output, cancellationToken);
            if (found is null) break;
            output.Add(specials[found]); position = next + found.Length;
        }
        return output.ToArray();
    }
    private void EncodeOrdinary(string text, List<int> output, CancellationToken cancellationToken)
    {
        var units = ("▁" + text.Replace(' ', '▁')).EnumerateRunes().Select(r => r.ToString()).ToList();
        while (units.Count > 1)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int best = int.MaxValue, index = -1;
            for (int i = 0; i + 1 < units.Count; i++) if (merges.TryGetValue((units[i], units[i + 1]), out int rank) && rank < best) { index = i; best = rank; }
            if (index < 0) break;
            units[index] += units[index + 1]; units.RemoveAt(index + 1);
        }
        foreach (string unit in units)
        {
            if (vocabulary.TryGetValue(unit, out int id)) output.Add(id);
            else foreach (byte b in Encoding.UTF8.GetBytes(unit)) output.Add(vocabulary["<0x" + b.ToString("X2", CultureInfo.InvariantCulture) + ">"]);
        }
    }
    internal int MaximumPieceCharacters => pieces.Max(p => p?.Length ?? 0);

    // Caller owns bounded scratch; only the final string handed to the caller is allocated.
    internal int DecodeInto(ReadOnlySpan<int> ids, Span<char> text, Span<byte> bytes)
    {
        int length = 0, pending = 0;
        foreach (int id in ids)
        {
            if ((uint)id >= pieces.Length || pieces[id] is not { } piece) throw new ArgumentOutOfRangeException(nameof(ids));
            if (specialIds.Contains(id)) continue;
            if (piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[^1] == '>' && byte.TryParse(piece.AsSpan(3, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out byte b))
            { bytes[pending++] = b; continue; }
            length += FlushBytes(bytes[..pending], text[length..]); pending = 0;
            foreach (char c in piece) text[length++] = c == '▁' ? ' ' : c;
        }
        length += FlushBytes(bytes[..pending], text[length..]);
        if (length > 0 && text[0] == ' ') { text.Slice(1, --length).CopyTo(text); }
        return length;
    }
    private static int FlushBytes(ReadOnlySpan<byte> bytes, Span<char> text)
    {
        if (bytes.IsEmpty) return 0;
        int offset = 0;
        while (offset < bytes.Length)
        {
            if (Rune.DecodeFromUtf8(bytes[offset..], out _, out int used) != System.Buffers.OperationStatus.Done)
            { text[..bytes.Length].Fill('\uFFFD'); return bytes.Length; }
            offset += used;
        }
        return Encoding.UTF8.GetChars(bytes, text);
    }

    /// <summary>Decodes valid vocabulary IDs, skipping special tokens by default. Replaces every byte in an invalid UTF-8 fallback run with U+FFFD. Removes one leading space per the tokenizer's Strip decoder. Throws ArgumentOutOfRangeException for missing/out-of-range IDs.</summary>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecialTokens = true, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = new StringBuilder();
        List<byte> pending = [];
        foreach (int id in ids)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if ((uint)id >= pieces.Length || pieces[id] is not { } piece) throw new ArgumentOutOfRangeException(nameof(ids));
            if (skipSpecialTokens && specialIds.Contains(id)) continue;
            if (piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[^1] == '>' && byte.TryParse(piece.AsSpan(3, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out byte b)) { pending.Add(b); continue; }
            Flush(); result.Append(piece.Replace('▁', ' '));
        }
        Flush();
        if (result.Length > 0 && result[0] == ' ') result.Remove(0, 1);
        return result.ToString();
        void Flush()
        {
            if (pending.Count == 0) return;
            // The released tokenizer replaces every byte in an invalid fallback run, including any valid prefix.
            try { result.Append(StrictUtf8.GetString(pending.ToArray())); }
            catch (DecoderFallbackException) { result.Append('\uFFFD', pending.Count); }
            pending.Clear();
        }
    }
}
