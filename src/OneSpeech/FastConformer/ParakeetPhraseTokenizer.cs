using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace OneSpeech.FastConformer;

// BPE ranks and normalization are read from the publisher's tokenizer data, not inferred from piece lengths.
internal sealed class ParakeetPhraseTokenizer
{
    private readonly Dictionary<string, int> vocabulary;
    private readonly Dictionary<(string, string), int> ranks = [];
    private readonly Dictionary<string, int> added = new(StringComparer.Ordinal);
    private readonly uint[] units;
    private readonly byte[] pool;
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);

    internal ParakeetPhraseTokenizer(JsonElement root, string[] pieces)
    {
        vocabulary = pieces.Select((piece, id) => (piece, id)).ToDictionary(pair => pair.piece, pair => pair.id, StringComparer.Ordinal);
        if (!vocabulary.TryGetValue("<unk>", out int unknown) || unknown != 0) throw new InvalidDataException("Phrase tokenizer requires unknown ID zero.");
        if (root.TryGetProperty("added_tokens", out var additions))
            foreach (var token in additions.EnumerateArray())
            {
                string content = token.GetProperty("content").GetString() ?? throw new InvalidDataException("Null added token.");
                if (content.Contains('<', StringComparison.Ordinal) || content.Contains('>', StringComparison.Ordinal)) continue;
                int id = token.GetProperty("id").GetInt32();
                if (content.Length is < 1 or > 1024 || added.Count == 4096 || token.GetProperty("special").GetBoolean() || token.GetProperty("normalized").GetBoolean() || token.GetProperty("lstrip").GetBoolean() || token.GetProperty("rstrip").GetBoolean() || token.GetProperty("single_word").GetBoolean() ||
                    !vocabulary.TryGetValue(content, out int expected) || expected != id || !added.TryAdd(content, id))
                    throw new InvalidDataException("Unsupported ordinary added-token contract.");
            }
        int rank = 0;
        foreach (var merge in root.GetProperty("model").GetProperty("merges").EnumerateArray())
        {
            if (rank >= 65536 || merge.GetArrayLength() != 2) throw new InvalidDataException("Invalid BPE merge table.");
            string a = merge[0].GetString()!, b = merge[1].GetString()!;
            if (!vocabulary.ContainsKey(a) || !vocabulary.ContainsKey(b) || !vocabulary.ContainsKey(a + b) || !ranks.TryAdd((a, b), rank++))
                throw new InvalidDataException("Invalid BPE merge pair.");
        }
        var normalizers = root.GetProperty("normalizer").GetProperty("normalizers");
        string map = normalizers[0].GetProperty("precompiled_charsmap").GetString()!;
        if (map.Length > 4 * 1024 * 1024) throw new InvalidDataException("Normalization map is too large.");
        byte[] data = Convert.FromBase64String(map);
        if (data.Length < 4) throw new InvalidDataException("Truncated normalization map.");
        uint bytes = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (bytes == 0 || bytes % 4 != 0 || bytes > data.Length - 4) throw new InvalidDataException("Invalid normalization trie size.");
        units = new uint[bytes / 4];
        for (int i = 0; i < units.Length; i++) units[i] = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(4 + i * 4));
        pool = data[(4 + (int)bytes)..];
    }

    internal ImmutableArray<int> Encode(string text, CancellationToken ct)
    {
        if (text.Contains('<', StringComparison.Ordinal) || text.Contains('>', StringComparison.Ordinal)) throw new ArgumentException("Control-token syntax is not allowed in contextual phrases.", nameof(text));
        // Non-normalized added tokens (the pinned model's raw digits) split the text before normalization.
        // Each remaining span gets its own prepend-always Metaspace prefix; normalized digits are not extracted again.
        var result = ImmutableArray.CreateBuilder<int>(); int cursor = 0;
        while (cursor < text.Length)
        {
            ct.ThrowIfCancellationRequested(); int position = text.Length, id = 0; string? match = null;
            foreach (var token in added)
            {
                int found = text.IndexOf(token.Key, cursor, StringComparison.Ordinal);
                if (found >= 0 && (found < position || found == position && token.Key.Length > (match?.Length ?? 0)))
                { position = found; match = token.Key; id = token.Value; }
            }
            result.AddRange(EncodeOrdinary(text[cursor..position], ct));
            if (match is null) break;
            result.Add(id); cursor = position + match.Length;
        }
        return result.ToImmutable();
    }

    private ImmutableArray<int> EncodeOrdinary(string text, CancellationToken ct)
    {
        string normalized = Normalize(text, ct).TrimEnd();
        // The pinned data declares Replace( {2,}, ▁), then prepend-always Metaspace with splitting.
        var collapsed = new StringBuilder();
        for (int i = 0; i < normalized.Length; i++)
        {
            if (normalized[i] == ' ' && i + 1 < normalized.Length && normalized[i + 1] == ' ')
            { collapsed.Append('▁'); while (i + 1 < normalized.Length && normalized[i + 1] == ' ') i++; }
            else collapsed.Append(normalized[i]);
        }
        normalized = collapsed.ToString().Replace(' ', '▁');
        if (normalized.Length == 0) return [];
        if (!normalized.StartsWith('▁')) normalized = "▁" + normalized;
        var output = ImmutableArray.CreateBuilder<int>();
        foreach (string word in normalized.Split('▁').Skip(1))
        {
            ct.ThrowIfCancellationRequested();
            var symbols = ("▁" + word).EnumerateRunes().Select(r => r.ToString()).ToList();
            while (symbols.Count > 1)
            {
                ct.ThrowIfCancellationRequested(); int best = int.MaxValue, position = -1;
                for (int i = 0; i + 1 < symbols.Count; i++)
                    if (ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < best) { best = rank; position = i; }
                if (position < 0) break;
                symbols[position] += symbols[position + 1]; symbols.RemoveAt(position + 1);
            }
            bool unknown = false;
            foreach (string symbol in symbols)
            {
                if (vocabulary.TryGetValue(symbol, out int id)) { output.Add(id); unknown = false; }
                else
                {
                    var fallback = StrictUtf8.GetBytes(symbol).Select(value => vocabulary.GetValueOrDefault("<0x" + value.ToString("X2", CultureInfo.InvariantCulture) + ">", -1)).ToArray();
                    if (fallback.All(value => value >= 0)) { output.AddRange(fallback); unknown = false; }
                    else { if (!unknown) output.Add(vocabulary["<unk>"]); unknown = true; }
                }
            }
        }
        return output.ToImmutable();
    }

    // Adapted from the family's OneNLP PrecompiledNormalizer. Offsets are not needed for context tokens.
    private string Normalize(string text, CancellationToken ct)
    {
        var result = new StringBuilder();
        var elements = StringInfo.GetTextElementEnumerator(text);
        while (elements.MoveNext())
        {
            ct.ThrowIfCancellationRequested(); string cluster = elements.GetTextElement();
            byte[] bytes = StrictUtf8.GetBytes(cluster);
            if (bytes.Length < 6 && TryReplace(bytes, out string replacement)) result.Append(replacement);
            else foreach (var rune in cluster.EnumerateRunes())
            {
                string value = rune.ToString();
                result.Append(TryReplace(StrictUtf8.GetBytes(value), out replacement) ? replacement : value);
            }
            if (result.Length > 8192) throw new InvalidDataException("Normalized phrase exceeds 8192 UTF-16 units.");
        }
        return result.ToString();
    }

    private bool TryReplace(byte[] key, out string replacement)
    {
        replacement = ""; int offset = -1; long node = Offset(units[0]);
        foreach (byte value in key)
        {
            node ^= value;
            if ((ulong)node >= (ulong)units.Length) break;
            uint unit = units[node];
            if ((unit & 0x800000FFu) != value) break;
            node ^= Offset(unit);
            if (((unit >> 8) & 1) != 0 && (ulong)node < (ulong)units.Length) offset = (int)(units[node] & 0x7FFFFFFFu);
        }
        if (offset < 0) return false;
        if (offset >= pool.Length) throw new InvalidDataException("Normalization replacement is outside its pool.");
        int end = Array.IndexOf(pool, (byte)0, offset);
        if (end < 0) throw new InvalidDataException("Unterminated normalization replacement.");
        try { replacement = StrictUtf8.GetString(pool.AsSpan(offset, end - offset)); }
        catch (DecoderFallbackException ex) { throw new InvalidDataException("Normalization replacement has invalid UTF-8.", ex); }
        return true;
    }
    private static long Offset(uint unit) => (unit >> 10) << (int)((unit & (1u << 9)) >> 6);
}
