using System.Buffers.Binary;
using System.Text.Json;
using OneSpeech.Formats;

namespace OneSpeech.Catalog.Conversion;

// Shared format-writing convention adapted from the Whisper converter. Tensor bytes are streamed at save time.
internal sealed class FastConformerGraph(IFloatTensorSource weights)
{
    private sealed record Value(string Name, int Dtype, object[] Shape);
    private sealed record Initializer(string Name, int Dtype, long[] Shape, byte[]? Bytes, string? Weight, long Offset, long Length);
    internal sealed record Attribute(string Name, int Type, long Integer = 0, float Float = 0, long[]? Integers = null);
    private sealed record Node(string Op, string[] Inputs, string[] Outputs, Attribute[] Attributes);
    private readonly List<Value> inputs = [];
    private readonly List<Value> outputs = [];
    private readonly List<Initializer> initializers = [];
    private readonly List<Node> nodes = [];
    private readonly HashSet<string> weightNames = new(StringComparer.Ordinal);
    private int serial;
    private long dataLength;

    private string AddTensor(string name, int dtype, long[] shape, byte[]? bytes, string? weight)
    {
        long length = bytes?.LongLength ?? checked(shape.Aggregate(1L, (a, b) => checked(a * b)) * 4);
        dataLength = (dataLength + 63) / 64 * 64;
        initializers.Add(new(name, dtype, shape, bytes, weight, dataLength, length));
        dataLength += length;
        return name;
    }

    internal void Input(string name, params object[] shape) => inputs.Add(new(name, 1, shape));
    internal void Output(string value, string name, params object[] shape)
    {
        nodes.Add(new("Identity", [value], [name], []));
        outputs.Add(new(name, 1, shape));
    }
    internal string Weight(string name, params long[] expectedShape)
    {
        if (!weights.GetShape(name).SequenceEqual(expectedShape)) throw new InvalidDataException($"Tensor '{name}' has the wrong shape.");
        if (weightNames.Add(name)) AddTensor(name, 1, expectedShape, null, name);
        return name;
    }
    internal string LongScalar(long value)
    {
        byte[] bytes = new byte[8]; BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
        return AddTensor("c" + serial++, 7, [], bytes, null);
    }
    internal string Ints(params long[] values)
    {
        byte[] bytes = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(i * 8), values[i]);
        return AddTensor("c" + serial++, 7, [values.Length], bytes, null);
    }
    internal string Float(float[] values, params long[] shape)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        return AddTensor("c" + serial++, 1, shape, bytes, null);
    }
    internal string Scalar(float value) => Float([value]);
    internal string Op(string op, string[] operands, params Attribute[] attributes)
    {
        string name = "v" + serial++;
        nodes.Add(new(op, operands, [name], attributes));
        return name;
    }
    internal static Attribute A(string name, long value) => new(name, 2, value);
    internal static Attribute As(string name, params long[] values) => new(name, 7, Integers: values);
    internal static Attribute F(string name, float value) => new(name, 1, Float: value);
    internal string Transpose(string x, params long[] permutation) => Op("Transpose", [x], As("perm", permutation));
    internal string Reshape(string x, params long[] shape) => Op("Reshape", [x, Ints(shape)]);
    internal string Slice(string x, int axis, int start, int end) => Op("Slice", [x, Ints(start), Ints(end), Ints(axis)]);
    internal string Linear(string x, string prefix, int input, int output, bool bias = false)
    {
        string y = Op("MatMul", [x, Transpose(Weight(prefix + ".weight", output, input), 1, 0)]);
        return bias ? Op("Add", [y, Weight(prefix + ".bias", output)]) : y;
    }
    internal string Norm(string x, string prefix, int width) => Op("LayerNormalization", [x, Weight(prefix + ".weight", width), Weight(prefix + ".bias", width)], A("axis", -1), F("epsilon", 1e-5f));
    internal string Swish(string x) => Op("Mul", [x, Op("Sigmoid", [x])]);
    internal void Save(string path, CancellationToken cancellationToken)
    {
        using var manifest = new MemoryStream();
        using (var json = new Utf8JsonWriter(manifest))
        {
            json.WriteStartObject(); json.WriteString("format", "onemodel"); json.WriteNumber("version", 1); json.WriteNumber("ir_version", 8); json.WriteString("producer_name", "OneSpeech");
            json.WriteStartArray("opsets"); json.WriteStartObject(); json.WriteString("domain", ""); json.WriteNumber("version", 17); json.WriteEndObject(); json.WriteEndArray();
            json.WriteStartObject("graph"); json.WriteString("name", "FastConformer"); WriteValues(json, "inputs", inputs); WriteValues(json, "outputs", outputs);
            json.WriteStartArray("initializers");
            foreach (var tensor in initializers)
            {
                json.WriteStartObject(); json.WriteString("name", tensor.Name); json.WriteNumber("dtype", tensor.Dtype); json.WriteStartArray("dims"); foreach (long dim in tensor.Shape) json.WriteNumberValue(dim); json.WriteEndArray();
                json.WriteStartObject("data"); json.WriteNumber("offset", tensor.Offset); json.WriteNumber("length", tensor.Length); json.WriteEndObject(); json.WriteEndObject();
            }
            json.WriteEndArray(); json.WriteStartArray("nodes");
            foreach (var node in nodes)
            {
                json.WriteStartObject(); json.WriteString("op", node.Op); WriteStrings(json, "inputs", node.Inputs); WriteStrings(json, "outputs", node.Outputs); json.WriteStartArray("attributes");
                foreach (var attr in node.Attributes)
                {
                    json.WriteStartObject(); json.WriteString("name", attr.Name); json.WriteNumber("type", attr.Type);
                    if (attr.Type == 1) json.WriteNumber("value", BitConverter.SingleToUInt32Bits(attr.Float));
                    else if (attr.Type == 2) json.WriteNumber("value", attr.Integer);
                    else { json.WriteStartArray("value"); foreach (long value in attr.Integers!) json.WriteNumberValue(value); json.WriteEndArray(); }
                    json.WriteEndObject();
                }
                json.WriteEndArray(); json.WriteEndObject();
            }
            json.WriteEndArray(); json.WriteEndObject(); json.WriteEndObject();
        }
        long dataOffset = (64 + manifest.Length + 63) / 64 * 64;
        byte[] header = new byte[64]; "ONEMODEL"u8.CopyTo(header); BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), 1);
        void Set(int offset, long value) => BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(offset), value);
        Set(16, 64); Set(24, manifest.Length); Set(32, dataOffset); Set(40, dataLength); Set(48, dataOffset + dataLength);
        if (dataOffset + dataLength >= int.MaxValue) throw new InvalidDataException("Split the graph into files below 2 GiB.");
        using FileStream file = File.Create(path); file.Write(header); manifest.Position = 0; manifest.CopyTo(file); file.SetLength(dataOffset + dataLength);
        foreach (var tensor in initializers)
        {
            cancellationToken.ThrowIfCancellationRequested(); file.Position = dataOffset + tensor.Offset;
            if (tensor.Bytes != null) file.Write(tensor.Bytes);
            else weights.CopyFloat32To(tensor.Weight!, file, cancellationToken);
        }
    }
    private static void WriteStrings(Utf8JsonWriter json, string key, IEnumerable<string> values)
    {
        json.WriteStartArray(key); foreach (string value in values) json.WriteStringValue(value); json.WriteEndArray();
    }
    private static void WriteValues(Utf8JsonWriter json, string key, IEnumerable<Value> values)
    {
        json.WriteStartArray(key);
        foreach (var value in values)
        {
            json.WriteStartObject(); json.WriteString("name", value.Name); json.WriteStartObject("type"); json.WriteString("kind", "tensor"); json.WriteNumber("elem", value.Dtype); json.WriteStartArray("shape");
            foreach (object dim in value.Shape) { if (dim is int number) json.WriteNumberValue(number); else json.WriteStringValue((string)dim); }
            json.WriteEndArray(); json.WriteEndObject(); json.WriteEndObject();
        }
        json.WriteEndArray();
    }
}
