using System.Text.Json;

namespace OneSpeech.Moonshine;

internal sealed record MoonshineConfiguration(int Width, int EncoderLayers, int DecoderLayers, int EncoderHeads, int DecoderHeads, int RotaryDimension, bool Streaming, int EncoderWidth, int EncoderHeadDimension, int DecoderHeadDimension)
{
    internal static MoonshineConfiguration Load(string directory)
    {
        string path = Path.Combine(directory, "config.json");
        using var input = File.OpenRead(path);
        long length = input.Length;
        if (length > 32768) throw new ModelIntegrityException("Moonshine config exceeds 32 KiB.");
        var bytes = new byte[checked((int)length)]; input.ReadExactly(bytes);
        return Parse(bytes);
    }
    internal static MoonshineConfiguration Parse(byte[] bytes)
    {
        if (bytes.Length > 32768) throw new ModelIntegrityException("Moonshine config exceeds 32 KiB.");
        using var json = JsonDocument.Parse(bytes, new JsonDocumentOptions { MaxDepth = 16 });
        try { return Read(json.RootElement); }
        catch (Exception error) when (error is KeyNotFoundException or InvalidOperationException or FormatException or OverflowException) { throw new ModelIntegrityException("Invalid Moonshine configuration structure.", error); }
    }
    private static MoonshineConfiguration Read(JsonElement c)
    {
        string? kind = c.GetProperty("model_type").GetString();
        if (kind is not ("moonshine" or "moonshine_streaming")) throw new ModelIntegrityException("Unsupported Moonshine configuration family.");
        bool streaming = kind == "moonshine_streaming";
        int width = c.GetProperty("hidden_size").GetInt32();
        int enc = streaming ? c.GetProperty("encoder_config").GetProperty("num_hidden_layers").GetInt32() : c.GetProperty("encoder_num_hidden_layers").GetInt32();
        int dec = c.GetProperty(streaming ? "num_hidden_layers" : "decoder_num_hidden_layers").GetInt32();
        int eh = streaming ? c.GetProperty("encoder_config").GetProperty("num_attention_heads").GetInt32() : c.GetProperty("encoder_num_attention_heads").GetInt32();
        int dh = c.GetProperty(streaming ? "num_attention_heads" : "decoder_num_attention_heads").GetInt32();
        double factor = streaming ? c.GetProperty("rope_parameters").GetProperty("partial_rotary_factor").GetDouble() : c.GetProperty("partial_rotary_factor").GetDouble();
        if (width is < 64 or > 2048 || eh is < 1 or > 32 || dh is < 1 or > 32 || !streaming && width % eh != 0 || width % dh != 0 || enc is < 1 or > 32 || dec is < 1 or > 32 || !double.IsFinite(factor) || factor <= 0 || factor > 1)
            throw new ModelIntegrityException("Moonshine configuration dimensions are outside supported bounds.");
        int ew = streaming ? c.GetProperty("encoder_config").GetProperty("hidden_size").GetInt32() : width;
        int ed = streaming ? c.GetProperty("encoder_config").GetProperty("head_dim").GetInt32() : width / eh;
        int dd = streaming ? c.GetProperty("head_dim").GetInt32() : width / dh;
        if (ew is < 64 or > 2048 || ed is < 8 or > 256 || dd is < 8 or > 256 || dd * dh != width) throw new ModelIntegrityException("Moonshine encoder/head dimensions are outside supported bounds.");
        int rotary = (int)(dd * factor) / 2 * 2;
        if (rotary < 2) throw new ModelIntegrityException("Moonshine rotary dimension must be positive and even.");
        return new(width, enc, dec, eh, dh, rotary, streaming, ew, ed, dd);
    }
}
