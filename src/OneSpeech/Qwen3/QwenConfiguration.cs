using System.Text.Json;

namespace OneSpeech.Qwen3;

internal sealed record QwenConfiguration(int AudioWidth, int AudioHeads, int AudioLayers, int DownsampleWidth, int Width, int Heads, int KvHeads, int HeadDim, int Layers, int Vocabulary, int TimestampClasses)
{
    internal static QwenConfiguration Load(string directory)
    {
        try
        {
            using var file = File.OpenRead(Path.Combine(directory, "config.json"));
            if (file.Length > 1048576) throw new InvalidDataException("Qwen configuration exceeds 1 MiB.");
            using var doc = JsonDocument.Parse(file, new JsonDocumentOptions { MaxDepth = 16 });
            var thinker = doc.RootElement.GetProperty("thinker_config");
            var audio = thinker.GetProperty("audio_config");
            var text = thinker.GetProperty("text_config");
            static int Get(JsonElement obj, string name, int limit)
            {
                int value = obj.GetProperty(name).GetInt32();
                if (value <= 0 || value > limit) throw new InvalidDataException("Unsupported Qwen dimension: " + name);
                return value;
            }
            if (audio.GetProperty("num_mel_bins").GetInt32() != 128 || audio.GetProperty("n_window").GetInt32() != 50 ||
                audio.GetProperty("n_window_infer").GetInt32() != 800 || audio.GetProperty("activation_function").GetString() != "gelu" ||
                text.GetProperty("hidden_act").GetString() != "silu" || text.GetProperty("rope_theta").GetDouble() != 1000000 ||
                text.GetProperty("rms_norm_eps").GetDouble() != 1e-6 || text.GetProperty("attention_bias").GetBoolean())
                throw new InvalidDataException("Qwen architecture parameters differ from the supported released checkpoints.");
            var result = new QwenConfiguration(Get(audio, "d_model", 2048), Get(audio, "encoder_attention_heads", 32), Get(audio, "encoder_layers", 64), Get(audio, "downsample_hidden_size", 1024),
                Get(text, "hidden_size", 4096), Get(text, "num_attention_heads", 64), Get(text, "num_key_value_heads", 64), Get(text, "head_dim", 256), Get(text, "num_hidden_layers", 64), Get(text, "vocab_size", 200000), thinker.TryGetProperty("classify_num", out var classes) ? classes.GetInt32() : 0);
            if (result.AudioWidth % result.AudioHeads != 0 || result.Heads % result.KvHeads != 0 || result.HeadDim % 2 != 0) throw new InvalidDataException("Unsupported Qwen attention shape.");
            return result;
        }
        catch (Exception ex) when (ex is InvalidOperationException or KeyNotFoundException or FormatException or OverflowException or System.Text.DecoderFallbackException)
        {
            throw new InvalidDataException("Invalid Qwen configuration data.", ex);
        }
    }
}
