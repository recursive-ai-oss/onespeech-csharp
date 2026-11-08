using System.Text.Json;
using OneSpeech.FastConformer;
using SharpFuzz;

if (args is ["--seed", var directory])
{
    Directory.CreateDirectory(directory);
    using var stream = File.Create(Path.Combine(directory, "vocabulary.json"));
    using var writer = new Utf8JsonWriter(stream);
    writer.WriteStartObject(); writer.WriteStartObject("model"); writer.WriteString("type", "BPE"); writer.WriteStartObject("vocab");
    for (int i = 0; i < 8192; i++) writer.WriteNumber("▁piece" + i, i);
    writer.WriteEndObject(); writer.WriteEndObject(); writer.WriteEndObject();
    return;
}
Fuzzer.LibFuzzer.Run(data =>
{
    try
    {
        var vocabulary = new ParakeetVocabulary(data.ToArray());
        _ = vocabulary.Words([new(0, 0, 1, .5f)]);
    }
    catch (InvalidDataException) { }
});
