using System.Buffers.Binary;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Vad;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class MoonshineAllocationProfile
{
    internal static void SileroOperators()
    {
        using var resource = typeof(SileroVad).Assembly.GetManifestResourceStream("OneSpeech.silero_vad.onnx")!;
        using var model = RecursiveAI.OneModel.Onnx.OnnxImporter.Load(resource);
        var counts = new SortedDictionary<string, int>(StringComparer.Ordinal);
        Visit(model.Graph);
        foreach (var pair in counts) Console.WriteLine(pair.Key + " " + pair.Value);
        void Visit(IrGraph graph)
        {
            foreach (var node in graph.Nodes)
            {
                counts[node.OpType] = counts.GetValueOrDefault(node.OpType) + 1;
                foreach (var attr in node.Attributes)
                {
                    if (attr.Type == IrAttributeType.Graph) Visit(attr.Graph);
                    if (attr.Type == IrAttributeType.Graphs) foreach (var child in attr.Graphs) Visit(child);
                }
            }
        }
    }
    internal static void Run(string? directory, string output)
    {
        using var file = File.Create(output);
        using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteString("status", "indicative allocation profile; fixed warmed shapes; one inference thread");
        json.WriteString("runtime", Environment.Version.ToString());
        json.WriteString("oneModel", "1.2.0-preview.2");
        string tiny = Path.Combine(Path.GetDirectoryName(output)!, "add-repro.onemodel");
        WriteAdd(tiny);
        using (var session = Session.Load(tiny, new ModelOptions { IntraOpThreads = 1 }))
        using (var input = Tensor.FromSpan<float>(new float[] { 1 }, Shape.Create(1)))
        {
            string[] inputNames = ["x"], outputNames = ["y"];
            Tensor[] inputs = [input], outputs = [Tensor.Zeros(DType.F32, Shape.Create(1))];
            try
            {
                Measure("addCallerBuffers", 10000, () => session.Run(inputNames, inputs, outputNames, outputs));
                if (outputs[0].Data<float>()[0] != 2) throw new InvalidOperationException("Add repro output mismatch.");
            }
            finally { outputs[0].Dispose(); }
        }
        string conditional = Path.Combine(Path.GetDirectoryName(output)!, "if-repro.onemodel");
        WriteIf(conditional);
        using (var session = Session.Load(conditional, new ModelOptions { IntraOpThreads = 1 }))
        using (var condition = Tensor.Scalar(true))
        using (var input = Tensor.FromSpan<float>(new float[] { 1 }, Shape.Create(1)))
        {
            string[] inputNames = ["condition", "x"], outputNames = ["y"];
            Tensor[] inputs = [condition, input], outputs = [Tensor.Zeros(DType.F32, Shape.Create(1))];
            try
            {
                Measure("ifCallerBuffers", 10000, () => session.Run(inputNames, inputs, outputNames, outputs));
                if (outputs[0].Data<float>()[0] != 2) throw new InvalidOperationException("If repro output mismatch.");
            }
            finally { outputs[0].Dispose(); }
        }
        if (directory is null) { json.WriteEndObject(); return; }
        using (var vad = new SileroVad())
        using (var state = vad.CreateState(16000))
        {
            float[] frame = new float[512];
            Measure("sileroCallerBuffers512", 128, () => state.ProcessFrame(frame, CancellationToken.None));
        }
        using (var events = new SnapshotSession())
        {
            var audio = new AudioBuffer(new float[1], 16000);
            Measure("immutablePartialSnapshot", 10000, () => events.Push(audio));
        }
        using var memory = Tensor.Zeros(DType.F32, Shape.Create(1, 164, 320));
        using var tokens = Tensor.FromSpan<long>(new long[] { 1, 500, 501, 502, 503, 504, 505, 506 }, Shape.Create(1, 8));
        int rotary = OneSpeech.Moonshine.MoonshineConfiguration.Load(directory).RotaryDimension;
        using var cos = OneSpeech.Moonshine.MoonshineModelSession.Rotary(8, rotary, false);
        using var sin = OneSpeech.Moonshine.MoonshineModelSession.Rotary(8, rotary, true);
        using var mask = Tensor.FromSpan<float>(Enumerable.Range(0, 64).Select(i => i % 8 > i / 8 ? float.NegativeInfinity : 0).ToArray(), Shape.Create(1, 1, 8, 8));
        using var positions = Tensor.FromSpan<long>(Enumerable.Range(0, 164).Select(i => (long)i).ToArray(), Shape.Create(164));
        using var decoder = Session.Load(Path.Combine(directory, "decoder.onemodel"), new ModelOptions { IntraOpThreads = 1, MaxCachedPlans = 2 });
        string[] names = ["tokens", "encoder", "cos", "sin", "mask", "encoder_positions"], wanted = ["logits"];
        Tensor[] decoderInputs = [tokens, memory, cos, sin, mask, positions], decoderOutputs = new Tensor[1];
        try
        {
            Measure("decoderCallerBuffers8Tokens164Frames", 32, () => decoder.Run(names, decoderInputs, wanted, decoderOutputs));
            var inputs = names.Select((n, i) => (n, i)).ToDictionary(v => v.n, v => decoderInputs[v.i], StringComparer.Ordinal);
            Measure("decoderContainer8Tokens164Frames", 32, () => { using var result = decoder.Run(inputs); });
        }
        finally { foreach (var tensor in decoderOutputs) tensor.Dispose(); }
        foreach (int capacity in new[] { 2, 16 })
        {
            using var changing = Session.Load(Path.Combine(directory, "decoder.onemodel"), new ModelOptions { IntraOpThreads = 1, MaxCachedPlans = capacity });
            var steps = new Tensor[16][];
            var results = new Tensor[16][];
            var owned = new List<Tensor>();
            try
            {
                for (int n = 1; n <= 16; n++)
                {
                    var token = Tensor.FromSpan<long>(Enumerable.Range(0, n).Select(i => i == 0 ? 1L : 500L + i).ToArray(), Shape.Create(1, n));
                    var cosine = OneSpeech.Moonshine.MoonshineModelSession.Rotary(n, rotary, false);
                    var sine = OneSpeech.Moonshine.MoonshineModelSession.Rotary(n, rotary, true);
                    var causal = Tensor.FromSpan<float>(Enumerable.Range(0, n * n).Select(i => i % n > i / n ? float.NegativeInfinity : 0).ToArray(), Shape.Create(1, 1, n, n));
                    owned.AddRange([token, cosine, sine, causal]);
                    steps[n - 1] = [token, memory, cosine, sine, causal, positions];
                    results[n - 1] = [Tensor.Zeros(DType.F32, Shape.Create(1, n, 32768))];
                }
                int index = 0;
                Measure("decoderVaryingPrefixCallerBuffersCache" + capacity, 32, () =>
                {
                    changing.Run(names, steps[index], wanted, results[index]);
                    index = (index + 1) % steps.Length;
                });
            }
            finally
            {
                foreach (var tensor in owned) tensor.Dispose();
                foreach (var result in results) if (result is not null) foreach (var tensor in result) tensor.Dispose();
            }
        }
        json.WriteEndObject();
        void Measure(string name, int iterations, Action action)
        {
            for (int i = 0; i < 20; i++) action();
            long before = GC.GetAllocatedBytesForCurrentThread();
            long total = GC.GetTotalAllocatedBytes(precise: true);
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < iterations; i++) action();
            double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            long totalBytes = GC.GetTotalAllocatedBytes(precise: true) - total;
            json.WriteStartObject(name); json.WriteNumber("calls", iterations); json.WriteNumber("threadBytesPerCall", bytes / (double)iterations); json.WriteNumber("totalBytesPerCall", totalBytes / (double)iterations); json.WriteNumber("secondsPerCall", seconds / iterations); json.WriteEndObject();
        }
    }

    private sealed class SnapshotSession : LiveSession
    {
        private readonly Segment segment = new() { Text = "x" };
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken) => PublishPartial(segment);
        protected override void Finish(CancellationToken cancellationToken) { }
    }

    // OneModel v1 format: a scalar Add with no weights, all buffers allocated by the caller.
    private static void WriteAdd(string path)
    {
        WriteModel(path, """
        {"format":"onemodel","version":1,"ir_version":8,"opsets":[{"domain":"","version":17}],"graph":{"name":"AddAllocationRepro","inputs":[{"name":"x","type":{"kind":"tensor","elem":1,"shape":[1]}}],"outputs":[{"name":"y","type":{"kind":"tensor","elem":1,"shape":[1]}}],"initializers":[],"nodes":[{"op":"Add","inputs":["x","x"],"outputs":["y"],"attributes":[]}]}}
        """);
    }

    private static void WriteIf(string path)
    {
        const string branch = """
        {"name":"branch","inputs":[],"outputs":[{"name":"out","type":{"kind":"tensor","elem":1,"shape":[1]}}],"initializers":[],"nodes":[{"op":"Add","inputs":["x","x"],"outputs":["out"],"attributes":[]}]}
        """;
        string graph = """
        {"format":"onemodel","version":1,"ir_version":8,"opsets":[{"domain":"","version":17}],"graph":{"name":"IfAllocationRepro","inputs":[{"name":"condition","type":{"kind":"tensor","elem":9,"shape":[]}},{"name":"x","type":{"kind":"tensor","elem":1,"shape":[1]}}],"outputs":[{"name":"y","type":{"kind":"tensor","elem":1,"shape":[1]}}],"initializers":[],"nodes":[{"op":"If","inputs":["condition"],"outputs":["y"],"attributes":[
        """ + "{\"name\":\"then_branch\",\"type\":5,\"value\":" + branch + "},{\"name\":\"else_branch\",\"type\":5,\"value\":" + branch + "}]}]}}";
        WriteModel(path, graph);
    }

    private static void WriteModel(string path, string json)
    {
        byte[] manifest = Encoding.UTF8.GetBytes(json);
        long end = (64 + manifest.Length + 63) / 64 * 64;
        byte[] header = new byte[64]; "ONEMODEL"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), 1);
        BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(16), 64);
        BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(24), manifest.Length);
        BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(32), end);
        BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(48), end);
        using var file = File.Create(path); file.Write(header); file.Write(manifest); file.SetLength(end);
    }
}
