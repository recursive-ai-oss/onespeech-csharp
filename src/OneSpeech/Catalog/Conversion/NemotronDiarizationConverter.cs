using OneSpeech.Formats;
using static OneSpeech.Catalog.Conversion.FastConformerGraph;

namespace OneSpeech.Catalog.Conversion;

// The pinned card/config specifies feature stacking and a pre-normalized Transformer, not Conformer blocks.
internal static class NemotronDiarizationConverter
{
    internal static void Convert(string input, string output, CancellationToken cancellationToken)
    {
        NemotronDiarizationAssets.VerifyPublisher(input, cancellationToken);
        using var file = new SafeTensorFile(File.OpenRead(Path.Combine(input, "model.safetensors")));
        var source = new SafeTensorSource(file);
        Directory.CreateDirectory(output);
        using (var silence = File.Create(Path.Combine(output, "silence.f32"))) source.CopyFloat32To("silence_embeds", silence, cancellationToken);
        var embed = new FastConformerGraph(source);
        embed.Input("stacked", 1, "frames", 1024);
        embed.Output(embed.Linear("stacked", "model.audio_tower.embedder.projection", 1024, 512), "embedded", 1, "frames", 512);
        embed.Save(Path.Combine(output, "embedding.onemodel"), cancellationToken);
        var graph = new FastConformerGraph(source);
        graph.Input("embedded", 1, "frames", 512);
        // RoPE phases are supplied by the caller: positions restart at zero for every bounded window.
        graph.Input("cos", 1, 1, "frames", 64);
        graph.Input("sin", 1, 1, "frames", 64);
        graph.Input("mask", 1, 1, 1, "frames");
        string x = graph.Norm("embedded", "model.audio_tower.input_layer_norm", 512);
        for (int layer = 0; layer < 31; layer++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string p = "model.audio_tower.layers." + layer;
            string y = graph.Norm(x, p + ".layer_norm1", 512);
            string q = Rope(Heads(graph.Linear(y, p + ".self_attn.q_proj", 512, 512)));
            string k = Rope(Heads(graph.Linear(y, p + ".self_attn.k_proj", 512, 512)));
            string v = Heads(graph.Linear(y, p + ".self_attn.v_proj", 512, 512));
            string scores = graph.Op("Div", [graph.Op("MatMul", [q, graph.Transpose(k, 0, 1, 3, 2)]), graph.Scalar(8)]);
            scores = graph.Op("Add", [scores, "mask"]);
            y = graph.Op("MatMul", [graph.Op("Softmax", [scores], A("axis", -1)), v]);
            y = graph.Reshape(graph.Transpose(y, 0, 2, 1, 3), 1, -1, 512);
            x = graph.Op("Add", [x, graph.Linear(y, p + ".self_attn.o_proj", 512, 512, bias: true)]);
            y = graph.Linear(graph.Norm(x, p + ".layer_norm2", 512), p + ".mlp.fc1", 512, 2048, bias: true);
            y = graph.Op("Mul", [graph.Op("Mul", [y, graph.Scalar(.5f)]), graph.Op("Add", [graph.Scalar(1), graph.Op("Erf", [graph.Op("Div", [y, graph.Scalar(MathF.Sqrt(2))])])])]);
            x = graph.Op("Add", [x, graph.Linear(y, p + ".mlp.fc2", 2048, 512, bias: true)]);
        }
        x = graph.Norm(x, "model.audio_tower.layer_norm", 512);
        graph.Output(x, "encoded", 1, "frames", 512);
        x = graph.Linear(x, "model.proj", 512, 192, bias: true);
        x = graph.Transpose(x, 0, 2, 1);
        x = graph.Op("Conv", [x, graph.Weight("model.upsampler.conv.weight", 1536, 192, 3), graph.Weight("model.upsampler.conv.bias", 1536)], As("pads", 1, 1));
        // Factor-major channel order and two ReLUs confirmed with bare-head public API probes.
        x = graph.Reshape(x, 1, 8, 192, -1);
        x = graph.Reshape(graph.Transpose(x, 0, 3, 1, 2), 1, -1, 192);
        graph.Output(x, "upsampled", 1, "mel_frames", 192);
        x = graph.Op("Relu", [graph.Linear(graph.Op("Relu", [x]), "classifier.dense", 192, 192, bias: true)]);
        x = graph.Linear(x, "classifier.out_proj", 192, 8, bias: true);
        graph.Output(graph.Op("Sigmoid", [x]), "probabilities", 1, "mel_frames", 8);
        graph.Save(Path.Combine(output, "diarization.onemodel"), cancellationToken);
        string Heads(string value) => graph.Transpose(graph.Reshape(value, 1, -1, 8, 64), 0, 2, 1, 3);
        string Rope(string value)
        {
            string rotated = graph.Op("Concat", [graph.Op("Neg", [graph.Slice(value, 3, 32, 64)]), graph.Slice(value, 3, 0, 32)], A("axis", 3));
            return graph.Op("Add", [graph.Op("Mul", [value, "cos"]), graph.Op("Mul", [rotated, "sin"])]);
        }
    }
}
