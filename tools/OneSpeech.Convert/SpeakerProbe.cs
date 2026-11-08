using System.Buffers.Binary;
using OneSpeech;
using OneSpeech.Embedders;

internal static class SpeakerProbe
{
    internal static void Run(string graph, string audioPath, string output)
    {
        var audio = OneAudio.Audio.Load(audioPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        var samples = audio.Samples.Span;
        Directory.CreateDirectory(output);
        float[] features = new SpeakerFbank().Compute(samples, CancellationToken.None);
        Write(Path.Combine(output, "features.f32"), features);
        if (graph == "features") return;
        var model = new ModelInfo { Id = "camp-en", Source = "https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k", Revision = "032b8131a7ad812f87061955ca974c99060c5a03", License = "Apache-2.0", Tier = ModelTier.Tier1, Attribution = "CAM++" };
        using var embedder = new CampPlusEmbedder(graph, model, threads: 8);
        float[] embedding = new float[embedder.Dimension];
        embedder.Embed(new AudioBuffer(audio.Samples, 16000), embedding, CancellationToken.None);
        Write(Path.Combine(output, "embedding.f32"), embedding);
        Console.WriteLine($"frames={features.Length / 80} dimension={embedding.Length}");
    }
    internal static void RunFeatures(string graphPath, string featurePath, string outputPath)
    {
        byte[] bytes = File.ReadAllBytes(featurePath);
        int frames = bytes.Length / (4 * 80);
        using var graph = RecursiveAI.OneModel.Graph.Session.Load(graphPath, new RecursiveAI.OneModel.Graph.ModelOptions { IntraOpThreads = 8 });
        using var input = RecursiveAI.OneModel.Tensors.Tensor.Zeros(RecursiveAI.OneModel.Tensors.DType.F32, RecursiveAI.OneModel.Tensors.Shape.Create(1, 80, frames));
        var data = input.MutableData<float>();
        for (int i = 0; i < data.Length; i++) data[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        using var outputs = graph.Run(new Dictionary<string, RecursiveAI.OneModel.Tensors.Tensor> { ["features"] = input });
        Write(outputPath, outputs["embedding"].Data<float>().ToArray());
    }
    private static void Write(string path, float[] values)
    {
        using var output = File.Create(path);
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        output.Write(bytes);
    }
}
