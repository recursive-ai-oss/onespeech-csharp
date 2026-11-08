using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

// Writes the published OneModel v1 format directly; all graph operations are standard ONNX opset 17.
internal sealed class SpeakerGraph
{
    internal sealed record Attribute(string Name, int Type, long Integer = 0, long[]? Integers = null, float Float = 0, string? Text = null);
    private sealed record Node(string Name, string Op, string[] Inputs, Attribute[] Attributes);
    private sealed record Constant(string Name, int Type, int[] Shape, byte[] Bytes);
    private readonly List<Node> nodes = [];
    private readonly List<Constant> constants = [];
    private int sequence;
    internal static Attribute I(string name, long value) => new(name, 2, Integer: value);
    internal static Attribute Is(string name, params long[] value) => new(name, 7, Integers: value);
    internal static Attribute F(string name, float value) => new(name, 1, Float: value);
    internal static Attribute S(string name, string value) => new(name, 3, Text: value);
    internal string Op(string op, string[] inputs, params Attribute[] attributes)
    {
        string name = "v" + sequence++;
        nodes.Add(new(name, op, inputs, attributes));
        return name;
    }
    internal string Float(string name, int[] shape, float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        constants.Add(new(name, 1, shape, bytes));
        return name;
    }
    internal string Long(params long[] values)
    {
        string name = "c" + sequence++;
        byte[] bytes = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(i * 8), values[i]);
        constants.Add(new(name, 7, [values.Length], bytes));
        return name;
    }
    internal string Reshape(string x, params long[] shape) => Op("Reshape", [x, Long(shape)]);
    internal string Mean(string x, params long[] axes) => Op("ReduceMean", [x], Is("axes", axes), I("keepdims", 1));
    internal string Concat(params string[] x) => Op("Concat", x, I("axis", 1));
    internal void Save(string path, string output, int dimension, string inputName = "features", object[]? inputShape = null, string outputName = "embedding", object[]? outputShape = null)
    {
        nodes.Add(new(outputName, "Identity", [output], []));
        long cursor = 0;
        using var manifest = new MemoryStream();
        using (var writer = new Utf8JsonWriter(manifest))
        {
            writer.WriteStartObject(); writer.WriteString("format", "onemodel"); writer.WriteNumber("version", 1); writer.WriteNumber("ir_version", 8);
            writer.WriteString("producer_name", "OneSpeech");
            writer.WriteStartArray("opsets"); writer.WriteStartObject(); writer.WriteString("domain", ""); writer.WriteNumber("version", 17); writer.WriteEndObject(); writer.WriteEndArray();
            writer.WriteStartObject("graph"); writer.WriteString("name", inputName == "waveform" ? "OmniAsrCtc" : "SpeakerEmbedding");
            writer.WriteStartArray("inputs"); Typed(writer, inputName, inputShape ?? [1, 80, "frames"]); writer.WriteEndArray();
            writer.WriteStartArray("outputs"); Typed(writer, outputName, outputShape ?? [1, dimension]); writer.WriteEndArray();
            writer.WriteStartArray("initializers");
            foreach (var c in constants)
            {
                cursor = Align(cursor);
                writer.WriteStartObject(); writer.WriteString("name", c.Name); writer.WriteNumber("dtype", c.Type);
                writer.WriteStartArray("dims"); foreach (int d in c.Shape) writer.WriteNumberValue(d); writer.WriteEndArray();
                writer.WriteStartObject("data"); writer.WriteNumber("offset", cursor); writer.WriteNumber("length", c.Bytes.Length); writer.WriteEndObject(); writer.WriteEndObject();
                cursor += c.Bytes.Length;
            }
            writer.WriteEndArray(); writer.WriteStartArray("nodes");
            foreach (var node in nodes)
            {
                writer.WriteStartObject(); writer.WriteString("name", node.Name); writer.WriteString("op", node.Op);
                writer.WriteStartArray("inputs"); foreach (string input in node.Inputs) writer.WriteStringValue(input); writer.WriteEndArray();
                writer.WriteStartArray("outputs"); writer.WriteStringValue(node.Name); writer.WriteEndArray();
                writer.WriteStartArray("attributes");
                foreach (var a in node.Attributes)
                {
                    writer.WriteStartObject(); writer.WriteString("name", a.Name); writer.WriteNumber("type", a.Type);
                    switch (a.Type)
                    {
                        case 1: writer.WriteNumber("value", BitConverter.SingleToUInt32Bits(a.Float)); break;
                        case 2: writer.WriteNumber("value", a.Integer); break;
                        case 3: writer.WriteString("value", Convert.ToBase64String(Encoding.UTF8.GetBytes(a.Text!))); break;
                        case 7: writer.WriteStartArray("value"); foreach (long n in a.Integers!) writer.WriteNumberValue(n); writer.WriteEndArray(); break;
                        default: throw new InvalidOperationException("Unknown graph attribute type.");
                    }
                    writer.WriteEndObject();
                }
                writer.WriteEndArray(); writer.WriteEndObject();
            }
            writer.WriteEndArray(); writer.WriteEndObject(); writer.WriteEndObject();
        }
        long dataOffset = Align(64 + manifest.Length), size = dataOffset + cursor;
        byte[] header = new byte[64];
        Encoding.ASCII.GetBytes("ONEMODEL").CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), 1);
        Write64(header, 16, 64); Write64(header, 24, manifest.Length); Write64(header, 32, dataOffset); Write64(header, 40, cursor); Write64(header, 48, size);
        using var file = File.Create(path); file.Write(header); manifest.Position = 0; manifest.CopyTo(file); Pad(file, dataOffset);
        cursor = 0;
        foreach (var c in constants) { cursor = Align(cursor); Pad(file, dataOffset + cursor); file.Write(c.Bytes); cursor += c.Bytes.Length; }
    }
    private static void Typed(Utf8JsonWriter writer, string name, object[] shape)
    {
        writer.WriteStartObject(); writer.WriteString("name", name); writer.WriteStartObject("type"); writer.WriteString("kind", "tensor"); writer.WriteNumber("elem", 1);
        writer.WriteStartArray("shape"); foreach (var d in shape) { if (d is int i) writer.WriteNumberValue(i); else writer.WriteStringValue((string)d); } writer.WriteEndArray();
        writer.WriteEndObject(); writer.WriteEndObject();
    }
    private static long Align(long n) => (n + 63) & ~63L;
    private static void Pad(Stream stream, long end) { while (stream.Position < end) stream.WriteByte(0); }
    private static void Write64(byte[] bytes, int offset, long value) => BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(offset), value);
}
