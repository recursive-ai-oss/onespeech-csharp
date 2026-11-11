using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using OneSpeech;
using OneSpeech.Embedders;

internal static class SpeakerBatch
{
    private static readonly string[] Splits = ["test-clean", "test-other"];
    internal static void PrepareIds(string trials, string clean, string allOutput, string parityOutput)
    {
        string[] parity = Directory.EnumerateFiles(clean, "*.flac", SearchOption.AllDirectories).Select(x => Path.GetFileNameWithoutExtension(x)).Order(StringComparer.Ordinal).Take(50).ToArray();
        var ids = SpeakerVerification.ReadTrials(trials).SelectMany(x => new[] { x.First, x.Second }).Concat(parity).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal);
        File.WriteAllLines(allOutput, ids); File.WriteAllLines(parityOutput, parity);
    }
    internal static void Extract(string graph, string data, string ids, string output, int start, int count, int dimension, bool ecapa = false, bool eres = false)
    {
        var files = Splits.SelectMany(split => Directory.EnumerateFiles(Path.Combine(data, split), "*.flac", SearchOption.AllDirectories)).ToDictionary(x => Path.GetFileNameWithoutExtension(x), StringComparer.Ordinal);
        var selected = File.ReadLines(ids).Skip(start).Take(count).ToArray();
        Directory.CreateDirectory(output);
        string modelId = ecapa ? "speechbrain/spkrec-ecapa-voxceleb" : eres ? "iic/speech_eres2netv2_sv_zh-cn_16k-common" : dimension == 512 ? "iic/speech_campplus_sv_en_voxceleb_16k" : "iic/speech_campplus_sv_zh-cn_16k-common";
        var metadata = new ModelInfo { Id = modelId, Source = (ecapa ? "https://huggingface.co/" : "https://modelscope.cn/models/") + modelId, Revision = ecapa ? "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286" : eres ? "1cf80d41fb3435bd3d8df185b5c423333b2db42a" : dimension == 512 ? "032b8131a7ad812f87061955ca974c99060c5a03" : "013c18884dffb82846cf472dd06dd4a08d97e2cd", Tier = ModelTier.Tier1, License = "Apache-2.0", Attribution = "Published checkpoint; retain Apache-2.0 licence and publisher notices." };
        var timer = Stopwatch.StartNew();
        using SpeakerEmbedder model = eres ? new ERes2NetV2Embedder(graph, metadata, threads: 8) : ecapa ? new EcapaTdnnEmbedder(graph, metadata, threads: 8) : new CampPlusEmbedder(graph, metadata, dimension, threads: 8);
        double load = timer.Elapsed.TotalSeconds, embeddingSeconds = 0, audioSeconds = 0;
        float[] vector = new float[dimension];
        var extractor = new SpeakerFbank();
        var ecapaExtractor = ecapa ? new EcapaFbank() : null;
        int completed = 0, skipped = 0;
        foreach (string id in selected)
        {
            string target = Path.Combine(output, id + ".f32");
            if (File.Exists(target) && new FileInfo(target).Length == dimension * 4) { skipped++; continue; }
            var clip = OneAudio.Audio.Load(files[id], new OneAudio.LoadOptions { Channels = 1, SampleRate = 16000 });
            timer.Restart();
            model.Embed(new AudioBuffer(clip.Samples, 16000), vector, CancellationToken.None);
            embeddingSeconds += timer.Elapsed.TotalSeconds; audioSeconds += (double)clip.Samples.Length / 16000;
            Write(target + ".partial", vector); File.Move(target + ".partial", target, overwrite: true);
            if (selected.Length <= 50) Write(Path.Combine(output, id + ".features.f32"), (ecapaExtractor is null ? extractor.Compute(clip.Samples.Span, CancellationToken.None) : ecapaExtractor.Compute(clip.Samples.Span, CancellationToken.None)));
            completed++;
            Console.WriteLine(id);
        }
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"indicative completed={completed} skipped={skipped} audioSeconds={audioSeconds:R} embeddingSeconds={embeddingSeconds:R} secondsPerAudioSecond={embeddingSeconds / audioSeconds:R} loadSeconds={load:R} threads=8 precision=fp32"));
    }
    private static void Write(string path, float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        File.WriteAllBytes(path, bytes);
    }
}
