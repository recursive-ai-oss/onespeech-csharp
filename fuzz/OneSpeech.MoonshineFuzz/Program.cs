using System.Globalization;
using System.Text.Json;
using OneSpeech.Moonshine;
using SharpFuzz;

if (args is ["--seed", var directory])
{
    Directory.CreateDirectory(directory);
    File.WriteAllText(Path.Combine(directory, "config"), "{\"model_type\":\"moonshine\",\"hidden_size\":64,\"encoder_num_hidden_layers\":1,\"decoder_num_hidden_layers\":1,\"encoder_num_attention_heads\":2,\"decoder_num_attention_heads\":2,\"partial_rotary_factor\":1}");
    using var file = File.Create(Path.Combine(directory, "tokenizer"));
    using var json = new Utf8JsonWriter(file);
    json.WriteStartObject();
    json.WritePropertyName("normalizer"); json.WriteRawValue("""{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]}""");
    json.WriteNull("pre_tokenizer");
    json.WritePropertyName("decoder"); json.WriteRawValue("""{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}""");
    json.WritePropertyName("post_processor"); json.WriteRawValue("""{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<s>"}},{"Sequence":{"id":"A"}}],"special_tokens":{"<s>":{"ids":[1]}}}""");
    json.WriteStartObject("model"); json.WriteString("type", "BPE"); json.WriteBoolean("byte_fallback", true); json.WriteStartObject("vocab");
    json.WriteNumber("<unk>", 0); json.WriteNumber("<s>", 1); json.WriteNumber("</s>", 2);
    for (int i = 0; i < 256; i++) json.WriteNumber("<0x" + i.ToString("X2", CultureInfo.InvariantCulture) + ">", i + 3);
    json.WriteNumber("▁", 259); json.WriteEndObject(); json.WriteStartArray("merges"); json.WriteEndArray(); json.WriteEndObject(); json.WriteStartArray("added_tokens");
    foreach (var (piece, id) in new[] { ("<unk>", 0), ("<s>", 1), ("</s>", 2) }) { json.WriteStartObject(); json.WriteString("content", piece); json.WriteNumber("id", id); json.WriteBoolean("special", true); json.WriteEndObject(); }
    json.WriteEndArray(); json.WriteEndObject();
    return;
}
Fuzzer.LibFuzzer.Run(data =>
{
    try
    {
        var tokenizer = new MoonshineTokenizer(data.ToArray());
        _ = tokenizer.Decode(tokenizer.Encode("a 🙂"));
    }
    catch (InvalidDataException) { }
    catch (JsonException) { }

    if (data.Length <= 32768)
        try { _ = MoonshineConfiguration.Parse(data.ToArray()); }
        catch (OneSpeech.ModelIntegrityException) { }
        catch (JsonException) { }
});
