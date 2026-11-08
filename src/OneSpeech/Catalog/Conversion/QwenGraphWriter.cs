using System.Buffers.Binary;
using System.Text.Json;
using OneSpeech.Formats;
using OneSpeech.Qwen3;

namespace OneSpeech.Catalog.Conversion;

// Qwen3-ASR §2.1: AuT convolutions, transformer blocks and audio projection.
internal sealed class QwenGraphWriter(QwenWeights weights, bool bfloat16 = false)
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
    private readonly Dictionary<string, string> weightCasts = new(StringComparer.Ordinal);
    private int serial;
    private long dataLength;

    private string AddTensor(string name, int dtype, long[] shape, byte[]? bytes, string? weight)
    {
        long length = bytes?.LongLength ?? checked(shape.Aggregate(1L, (a, b) => checked(a * b)) * (dtype == 16 ? 2 : 4));
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
            AddTensor(name, bfloat16 && info.DataType == "BF16" ? 16 : 1, info.Shape.ToArray(), null, name);
        }
        if (bfloat16 && weights.Tensors[name].DataType == "BF16")
        {
            if (!weightCasts.TryGetValue(name, out string? expanded))
            {
                expanded = Op("Cast", [name], A("to", 1)); weightCasts.Add(name, expanded);
            }
            return expanded;
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
    private string Round(string x) => bfloat16 ? Op("Cast", [Op("Cast", [x], A("to", 16))], A("to", 1)) : x;
    private static Attribute A(string name, long value) => new(name, 2, value);
    private static Attribute As(string name, params long[] values) => new(name, 7, Integers: values);
    private string Transpose(string x, params long[] permutation) => Op("Transpose", [x], As("perm", permutation));
    private string Reshape(string x, params long[] shape) => Op("Reshape", [x, Ints(shape)]);
    private string Linear(string x, string prefix, bool bias = true)
    {
        string y = Op("MatMul", [x, Transpose(Weight(prefix + ".weight"), 1, 0)]);
        return Round(bias ? Op("Add", [y, Weight(prefix + ".bias")]) : y);
    }
    private string Norm(string x, string prefix) => Round(Op("LayerNormalization", [x, Weight(prefix + ".weight"), Weight(prefix + ".bias")], A("axis", -1), new("epsilon", 1, Float: 1e-5f)));
    private string Gelu(string x) => Round(Op("Mul", [Op("Mul", [x, Scalar(.5f)]), Op("Add", [Scalar(1), Op("Erf", [Op("Div", [x, Scalar(MathF.Sqrt(2))])])])]));
    private string Heads(string x, int count, int dimension) => Transpose(Reshape(x, 1, -1, count, dimension), 0, 2, 1, 3);
    private (string Result, string Probabilities) Attend(string q, string k, string v, int heads, int width, string? mask = null)
    {
        // Attention keeps score/softmax intermediates in binary32 and rounds the attended output.
        // The published CPU PyTorch attention API is closer to this than to staged half-precision scores.
        float scale = (float)Math.Pow(width / heads, -.25);
        string scores = bfloat16
            ? Op("Mul", [Op("MatMul", [q, Transpose(k, 0, 1, 3, 2)]), Scalar(1 / MathF.Sqrt(width / heads))])
            : Op("MatMul", [Op("Mul", [q, Scalar(scale)]), Transpose(Op("Mul", [k, Scalar(scale)]), 0, 1, 3, 2)]);
        if (mask != null) scores = Op("Add", [scores, mask]);
        string probabilities = Op("Softmax", [scores], A("axis", -1));
        string result = Reshape(Transpose(Round(Op("MatMul", [probabilities, v])), 0, 2, 1, 3), 1, -1, width);
        return (result, probabilities);
    }
    private string FeedForward(string x, string prefix)
    {
        string n = Norm(x, prefix + ".final_layer_norm");
        return Round(Op("Add", [x, Linear(Gelu(Linear(n, prefix + ".fc1")), prefix + ".fc2")]));
    }
    private void Input(string name, int dtype, params object[] shape) => inputs.Add(new(name, dtype, shape));
    private void Output(string x, string name, params object[] shape)
    {
        nodes.Add(new("Identity", [x], [name], []));
        outputs.Add(new(name, 1, shape));
    }

    internal void EncoderFront(QwenConfiguration config)
    {
        Input("mel_chunks", 1, "chunks", 1, 128, 100);
        string x = "mel_chunks";
        for (int i = 1; i <= 3; i++)
            x = Gelu(Round(Op("Conv", [x, Weight("thinker.audio_tower.conv2d" + i + ".weight"), Weight("thinker.audio_tower.conv2d" + i + ".bias")], As("strides", 2, 2), As("pads", 1, 1, 1, 1))));
        x = Reshape(Transpose(x, 0, 3, 1, 2), -1, 13, config.DownsampleWidth * 16);
        Output(Linear(x, "thinker.audio_tower.conv_out", false), "conv", "chunks", 13, config.AudioWidth);
    }
    internal void Encoder(QwenConfiguration config)
    {
        Input("hidden", 1, 1, "sequence", config.AudioWidth);
        Input("positions", 1, 1, "sequence", config.AudioWidth);
        Input("mask", 1, 1, 1, "sequence", "sequence");
        string x = Round(Op("Add", ["hidden", "positions"]));
        for (int layer = 0; layer < config.AudioLayers; layer++)
        {
            string prefix = "thinker.audio_tower.layers." + layer;
            string n = Norm(x, prefix + ".self_attn_layer_norm");
            int dim = config.AudioWidth / config.AudioHeads;
            string q = Heads(Linear(n, prefix + ".self_attn.q_proj"), config.AudioHeads, dim);
            string k = Heads(Linear(n, prefix + ".self_attn.k_proj"), config.AudioHeads, dim);
            string v = Heads(Linear(n, prefix + ".self_attn.v_proj"), config.AudioHeads, dim);
            x = Round(Op("Add", [x, Linear(Attend(q, k, v, config.AudioHeads, config.AudioWidth, "mask").Result, prefix + ".self_attn.out_proj")]));
            x = FeedForward(x, prefix);
        }
        string post = Norm(x, "thinker.audio_tower.ln_post");
        Output(post, "normalized", 1, "sequence", config.AudioWidth);
        Output(Linear(Gelu(Linear(post, "thinker.audio_tower.proj1")), "thinker.audio_tower.proj2"), "encoder", 1, "sequence", config.Width);
    }
    private string RmsNorm(string x, string prefix)
    {
        string variance = Op("ReduceMean", [Op("Mul", [x, x])], As("axes", -1), A("keepdims", 1));
        return Round(Op("Mul", [Round(Op("Div", [x, Op("Sqrt", [Op("Add", [variance, Scalar(1e-6f)])])])), Weight(prefix + ".weight")]));
    }
    private string Rotary(string x, int dimension)
    {
        string left = Op("Slice", [x, Ints(0), Ints(dimension / 2), Ints(3)]);
        string right = Op("Slice", [x, Ints(dimension / 2), Ints(dimension), Ints(3)]);
        string rotated = Op("Concat", [Op("Neg", [right]), left], A("axis", 3));
        return Round(Op("Add", [Round(Op("Mul", [x, "cos"])), Round(Op("Mul", [rotated, "sin"]))]));
    }
    internal void Embeddings(QwenConfiguration config)
    {
        Input("tokens", 7, 1, "sequence");
        Input("audio", 1, 1, "audio_sequence", config.Width);
        Input("audio_positions", 7, "audio_sequence", 2);
        string tokens = Op("Gather", [Weight("thinker.model.embed_tokens.weight"), "tokens"], A("axis", 0));
        string audio = Op("Squeeze", ["audio", Ints(0)]);
        Output(Op("ScatterND", [tokens, "audio_positions", audio]), "embeddings", 1, "sequence", config.Width);
    }
    internal void LanguageHead(QwenConfiguration config)
    {
        Input("hidden", 1, 1, "sequence", config.Width);
        string last = config.TimestampClasses > 0 ? "hidden" : Op("Gather", ["hidden", Ints(-1)], A("axis", 1));
        Output(Linear(last, "thinker.lm_head", false), "logits", 1, config.TimestampClasses > 0 ? "sequence" : 1, config.TimestampClasses > 0 ? config.TimestampClasses : config.Vocabulary);
    }
    internal void Decoder(QwenConfiguration config, int begin, int end)
    {
        Input("hidden", 1, 1, "sequence", config.Width);
        Input("mask", 1, 1, 1, "sequence", "total");
        Input("cos", 1, 1, 1, "sequence", config.HeadDim);
        Input("sin", 1, 1, 1, "sequence", config.HeadDim);
        string x = "hidden";
        for (int layer = begin; layer < end; layer++)
        {
            string prefix = "thinker.model.layers." + layer;
            Input("past_k_" + layer, 1, 1, config.KvHeads, "past", config.HeadDim);
            Input("past_v_" + layer, 1, 1, config.KvHeads, "past", config.HeadDim);
            string n = RmsNorm(x, prefix + ".input_layernorm");
            string q = Rotary(RmsNorm(Heads(Linear(n, prefix + ".self_attn.q_proj", false), config.Heads, config.HeadDim), prefix + ".self_attn.q_norm"), config.HeadDim);
            string k = Rotary(RmsNorm(Heads(Linear(n, prefix + ".self_attn.k_proj", false), config.KvHeads, config.HeadDim), prefix + ".self_attn.k_norm"), config.HeadDim);
            string v = Heads(Linear(n, prefix + ".self_attn.v_proj", false), config.KvHeads, config.HeadDim);
            k = Op("Concat", ["past_k_" + layer, k], A("axis", 2));
            v = Op("Concat", ["past_v_" + layer, v], A("axis", 2));
            Output(k, "present_k_" + layer, 1, config.KvHeads, "total", config.HeadDim);
            Output(v, "present_v_" + layer, 1, config.KvHeads, "total", config.HeadDim);
            string ExpandHeads(string tensor)
            {
                if (config.Heads == config.KvHeads) return tensor;
                string expanded = Op("Unsqueeze", [tensor, Ints(2)]);
                expanded = Op("Tile", [expanded, Ints(1, 1, config.Heads / config.KvHeads, 1, 1)]);
                return Reshape(expanded, 1, config.Heads, -1, config.HeadDim);
            }
            string scores = Op("Mul", [Op("MatMul", [q, Transpose(ExpandHeads(k), 0, 1, 3, 2)]), Scalar(1 / MathF.Sqrt(config.HeadDim))]);
            scores = Op("Softmax", [Op("Add", [scores, "mask"])], A("axis", -1));
            string attention = Reshape(Transpose(Round(Op("MatMul", [scores, ExpandHeads(v)])), 0, 2, 1, 3), 1, -1, config.Heads * config.HeadDim);
            x = Round(Op("Add", [x, Linear(attention, prefix + ".self_attn.o_proj", false)]));
            n = RmsNorm(x, prefix + ".post_attention_layernorm");
            string gate = Linear(n, prefix + ".mlp.gate_proj", false);
            gate = Round(Op("Mul", [gate, Op("Sigmoid", [gate])]));
            x = Round(Op("Add", [x, Linear(Round(Op("Mul", [gate, Linear(n, prefix + ".mlp.up_proj", false)])), prefix + ".mlp.down_proj", false)]));
        }
        if (end == config.Layers) x = RmsNorm(x, "thinker.model.norm");
        Output(x, "decoder", 1, "sequence", config.Width);
    }
    internal void Save(string path, CancellationToken cancellationToken)
    {
        using var manifest = new MemoryStream();
        using (var json = new Utf8JsonWriter(manifest))
        {
            json.WriteStartObject(); json.WriteString("format", "onemodel"); json.WriteNumber("version", 1); json.WriteNumber("ir_version", 8); json.WriteString("producer_name", "OneSpeech");
            json.WriteStartArray("opsets"); json.WriteStartObject(); json.WriteString("domain", ""); json.WriteNumber("version", 17); json.WriteEndObject(); json.WriteEndArray();
            json.WriteStartObject("graph"); json.WriteString("name", "Qwen3"); WriteValues(json, "inputs", inputs); WriteValues(json, "outputs", outputs);
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
        if (dataOffset + dataLength > int.MaxValue) throw new CapabilityNotSupportedException("Qwen graph exceeds the OneModel file-size limit; split the graph further.");
        byte[] header = new byte[64]; "ONEMODEL"u8.CopyTo(header); BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), 1);
        void Set(int offset, long value) => BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(offset), value);
        Set(16, 64); Set(24, manifest.Length); Set(32, dataOffset); Set(40, dataLength); Set(48, dataOffset + dataLength);
        using FileStream file = File.Create(path); file.Write(header); manifest.Position = 0; manifest.CopyTo(file); file.SetLength(dataOffset + dataLength);
        foreach (var tensor in initializers)
        {
            cancellationToken.ThrowIfCancellationRequested(); file.Position = dataOffset + tensor.Offset;
            if (tensor.Bytes != null) file.Write(tensor.Bytes);
            else if (tensor.Dtype == 16) weights.CopyTensorTo(tensor.Weight!, file, cancellationToken);
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
