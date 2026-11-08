using System.Buffers.Binary;
using System.Text.Json;
using OneSpeech.Formats;
using OneSpeech.Moonshine;

namespace OneSpeech.Catalog.Conversion;

// The graph follows Jeffries et al. 2024 §3.1, with names taken only from published tensor data.
internal sealed partial class MoonshineGraphWriter(SafeTensorFile weights, bool diagnostics = false, int compressionMode = 0, int frontendActivation = 1, float frontendEpsilon = 1e-6f, int frontendNorm = 0, bool int8 = false, CancellationToken conversionCancellation = default)
{
    private sealed record Value(string Name, int Dtype, object[] Shape);
    private sealed record Initializer(string Name, int Dtype, long[] Shape, byte[]? Bytes, string? Weight, long Offset, long Length);
    private sealed record Attribute(string Name, int Type, long Integer = 0, float Float = 0, long[]? Integers = null);
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
    private string Weight(string name)
    {
        if (weightNames.Add(name))
        {
            SafeTensorInfo info = weights.Tensors[name];
            AddTensor(name, 1, info.Shape.ToArray(), null, name);
        }
        return name;
    }
    private string Ints(params long[] values)
    {
        byte[] bytes = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(i * 8), values[i]);
        return AddTensor("c" + serial++, 7, [values.Length], bytes, null);
    }
    private string Scalar(float value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return AddTensor("c" + serial++, 1, [], bytes, null);
    }
    private string Op(string op, string[] operands, params Attribute[] attributes)
    {
        string name = "v" + serial++;
        nodes.Add(new(op, operands, [name], attributes));
        return name;
    }
    private static Attribute A(string name, long value) => new(name, 2, value);
    private static Attribute As(string name, params long[] values) => new(name, 7, Integers: values);
    private string Transpose(string x, params long[] permutation) => Op("Transpose", [x], As("perm", permutation));
    private string Reshape(string x, params long[] shape) => Op("Reshape", [x, Ints(shape)]);
    private string Linear(string x, string prefix, bool bias = true)
    {
        string y = int8 ? Int8Linear(x, prefix + ".weight") : Op("MatMul", [x, Transpose(Weight(prefix + ".weight"), 1, 0)]);
        return bias ? Op("Add", [y, Weight(prefix + ".bias")]) : y;
    }
    private readonly Dictionary<string, (string Weight, string Scale)> quantizedWeights = new(StringComparer.Ordinal);
    private string Int8Linear(string x, string weight)
    {
        if (!quantizedWeights.TryGetValue(weight, out var packed))
        {
            var shape = weights.Tensors[weight].Shape;
            if (shape.Length != 2) throw new ModelIntegrityException("Expected rank-two linear weight.");
            var quantized = Int8Weights.Transpose(weights.ReadFloat32(weight, conversionCancellation), checked((int)shape[0]), checked((int)shape[1]), conversionCancellation);
            packed = (AddTensor(weight + ".int8", 3, [shape[1], shape[0]], quantized.Values, null),
                AddTensor(weight + ".scale", 1, [shape[0]], quantized.Scales, null));
            quantizedWeights.Add(weight, packed);
        }
        string q = "v" + serial++, scale = "v" + serial++, zero = "v" + serial++;
        nodes.Add(new("DynamicQuantizeLinear", [x], [q, scale, zero], []));
        string product = Op("MatMulInteger", [q, packed.Weight, zero, AddTensor("c" + serial++, 3, [], [0], null)]);
        return Op("Mul", [Op("Cast", [product], A("to", 1)), Op("Mul", [scale, packed.Scale])]);
    }
    private string Norm(string x, string prefix) => Op("LayerNormalization", [x, Weight(prefix + ".weight")], A("axis", -1), new("epsilon", 1, Float: 1e-5f));
    private string Gelu(string x) => Op("Mul", [Op("Mul", [x, Scalar(.5f)]), Op("Add", [Scalar(1), Op("Erf", [Op("Div", [x, Scalar(MathF.Sqrt(2))])])])]);
    private string Heads(string x, int count, int dimension) => Transpose(Reshape(x, 1, -1, count, dimension), 0, 2, 1, 3);
    private (string Result, string Probabilities) Attend(string q, string k, string v, int heads, int width, string? mask = null)
    {
        float scale = (float)Math.Pow(width / heads, -.25);
        string scores = Op("MatMul", [Op("Mul", [q, Scalar(scale)]), Transpose(Op("Mul", [k, Scalar(scale)]), 0, 1, 3, 2)]);
        if (mask != null) scores = Op("Add", [scores, mask]);
        string probabilities = Op("Softmax", [scores], A("axis", -1));
        string result = Reshape(Transpose(Op("MatMul", [probabilities, v]), 0, 2, 1, 3), 1, -1, width);
        return (result, probabilities);
    }
    private void Input(string name, int dtype, params object[] shape) => inputs.Add(new(name, dtype, shape));
    private void Output(string x, string name, params object[] shape)
    {
        nodes.Add(new("Identity", [x], [name], []));
        outputs.Add(new(name, 1, shape));
    }

    internal void Save(string path, CancellationToken cancellationToken)
    {
        using var manifest = new MemoryStream();
        using (var json = new Utf8JsonWriter(manifest))
        {
            json.WriteStartObject(); json.WriteString("format", "onemodel"); json.WriteNumber("version", 1); json.WriteNumber("ir_version", 8); json.WriteString("producer_name", "OneSpeech");
            json.WriteStartArray("opsets"); json.WriteStartObject(); json.WriteString("domain", ""); json.WriteNumber("version", 17); json.WriteEndObject(); json.WriteEndArray();
            json.WriteStartObject("graph"); json.WriteString("name", "Moonshine"); WriteValues(json, "inputs", inputs); WriteValues(json, "outputs", outputs);
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
        using FileStream file = File.Create(path); file.Write(header); manifest.Position = 0; manifest.CopyTo(file); file.SetLength(dataOffset + dataLength);
        foreach (var tensor in initializers)
        {
            cancellationToken.ThrowIfCancellationRequested(); file.Position = dataOffset + tensor.Offset;
            if (tensor.Bytes != null) file.Write(tensor.Bytes);
            else
            {
                if (weights.Tensors[tensor.Weight!].DataType == "F32") weights.CopyTensorTo(tensor.Weight!, file, cancellationToken);
                else
                {
                    float[] values = weights.ReadFloat32(tensor.Weight!, cancellationToken);
                    byte[] buffer = new byte[65536];
                    for (int i = 0; i < values.Length;)
                    {
                        int count = Math.Min(values.Length - i, buffer.Length / 4);
                        for (int j = 0; j < count; j++) BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(j * 4), values[i + j]);
                        file.Write(buffer, 0, count * 4); i += count;
                    }
                }
            }
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
