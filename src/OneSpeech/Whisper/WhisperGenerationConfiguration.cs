using System.Text.Json;

namespace OneSpeech.Whisper;

internal sealed record WhisperGenerationConfiguration(int Start, int End, int Transcribe, int Translate, int NoTimestamps,
    Dictionary<string, int> Languages, int[] Suppress, (int Layer, int Head)[] AlignmentHeads)
{
    internal static WhisperGenerationConfiguration Load(string directory, WhisperConfiguration configuration)
    {
        using var document = WhisperConfiguration.ReadJson(Path.Combine(directory, "generation_config.json"));
        try
        {
            var root = document.RootElement;
            int Token(JsonElement value)
            {
                int id = value.GetInt32();
                if (id < 0 || id >= configuration.Vocabulary) throw new InvalidDataException("Generation token ID exceeds the vocabulary.");
                return id;
            }
            int start = Token(root.GetProperty("decoder_start_token_id")), end = Token(root.GetProperty("eos_token_id"));
            var tasks = root.GetProperty("task_to_id");
            int transcribe = Token(tasks.GetProperty("transcribe")), translate = Token(tasks.GetProperty("translate"));
            int noTimestamps = Token(root.GetProperty("no_timestamps_token_id"));
            if (noTimestamps >= configuration.Vocabulary - 1) throw new InvalidDataException("No timestamp-token range remains.");
            var languages = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var entry in root.GetProperty("lang_to_id").EnumerateObject())
            {
                if (languages.Count >= 256 || entry.Name.Length is < 5 or > 32 || !entry.Name.StartsWith("<|", StringComparison.Ordinal) || !entry.Name.EndsWith("|>", StringComparison.Ordinal))
                    throw new InvalidDataException("Invalid language-token name or count.");
                if (!languages.TryAdd(entry.Name[2..^2], Token(entry.Value))) throw new InvalidDataException("Duplicate language-token name.");
            }
            var suppressed = root.GetProperty("suppress_tokens");
            if (suppressed.GetArrayLength() > configuration.Vocabulary) throw new InvalidDataException("Too many suppressed tokens.");
            int[] suppress = suppressed.EnumerateArray().Select(value => value.GetInt32() == -1 ? -1 : Token(value)).ToArray();
            (int Layer, int Head)[] heads = [];
            if (root.TryGetProperty("alignment_heads", out var selected))
            {
                if (selected.GetArrayLength() > configuration.DecoderLayers * configuration.DecoderHeads) throw new InvalidDataException("Too many alignment heads.");
                heads = selected.EnumerateArray().Select(value =>
                {
                    if (value.GetArrayLength() != 2) throw new InvalidDataException("Alignment heads require layer/head pairs.");
                    int layer = value[0].GetInt32(), head = value[1].GetInt32();
                    if (layer < 0 || layer >= configuration.DecoderLayers || head < 0 || head >= configuration.DecoderHeads) throw new InvalidDataException("Alignment head exceeds decoder dimensions.");
                    return (layer, head);
                }).ToArray();
            }
            return new(start, end, transcribe, translate, noTimestamps, languages, suppress, heads);
        }
        catch (Exception ex) when (ex is InvalidOperationException or KeyNotFoundException or FormatException or OverflowException or ArgumentException)
        { throw new InvalidDataException("Invalid Whisper generation configuration.", ex); }
    }
}
