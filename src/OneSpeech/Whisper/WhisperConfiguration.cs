using System.Text.Json;

namespace OneSpeech.Whisper;

internal sealed record WhisperConfiguration(int Width, int EncoderLayers, int DecoderLayers, int EncoderHeads, int DecoderHeads, int MelBins, int Vocabulary, int EncoderFfn, int DecoderFfn)
{
    internal static JsonDocument ReadJson(string path)
    {
        using FileStream file = File.OpenRead(path);
        if (file.Length > 1024 * 1024) throw new InvalidDataException("Whisper configuration exceeds 1 MiB.");
        return JsonDocument.Parse(file, new JsonDocumentOptions { MaxDepth = 16 });
    }

    internal static WhisperConfiguration Load(string directory)
    {
        using var doc = ReadJson(Path.Combine(directory, "config.json"));
        var root = doc.RootElement;
        int Get(string key, int max)
        {
            int value = root.GetProperty(key).GetInt32();
            if (value <= 0 || value > max) throw new InvalidDataException("Whisper configuration out of bounds: " + key);
            return value;
        }
        var config = new WhisperConfiguration(Get("d_model", 2048), Get("encoder_layers", 64), Get("decoder_layers", 64), Get("encoder_attention_heads", 32), Get("decoder_attention_heads", 32), Get("num_mel_bins", 128), Get("vocab_size", 60000), Get("encoder_ffn_dim", 8192), Get("decoder_ffn_dim", 8192));
        if (config.MelBins is not (80 or 128) || config.Width % config.EncoderHeads != 0 || config.Width % config.DecoderHeads != 0) throw new InvalidDataException("Unsupported Whisper dimensions.");
        return config;
    }
}
