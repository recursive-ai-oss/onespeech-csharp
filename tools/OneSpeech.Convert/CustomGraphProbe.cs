using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Text.Json;
using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.Custom;
using OneSpeech.FastConformer;
using OneSpeech.Text;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class CustomGraphProbe
{
    internal static void Fixture(string path)
    {
        var graph = new OneSpeech.Catalog.Conversion.FastConformerGraph(new NoWeights());
        graph.Input("input", 1, 1, "samples");
        graph.Output(graph.Float([9, -9, -9, -9, -9, 9, -9, 9, -9, -9, -9, 9], 1, 4, 3), "output", 1, 4, 3);
        graph.Save(path, CancellationToken.None);
    }
    internal static void Run(string kind, string directory, string corpus, string output)
    {
        Directory.CreateDirectory(output);
        var entry = ModelCatalog.Default.Get(kind == "ctc" ? "facebook/omniASR-CTC-300M" : "nvidia/parakeet-tdt-0.6b-v3");
        var info = new ModelInfo { Id = entry.Id, Revision = entry.Revision, Source = entry.Source, License = entry.License, Attribution = entry.Attribution, Tier = entry.Tier };
        TokenVocabulary vocabulary;
        GraphSpeechRecognizer custom;
        Session? ctc = null; ParakeetRecognizer? tdt = null;
        if (kind == "ctc")
        {
            var data = SentencePieceProto.ReadFile(Path.Combine(directory, "omniASR_tokenizer.model"));
            vocabulary = new(data.Pieces.Select((piece, i) => data.Types[i] == 1 ? piece : ""));
            string graph = Path.Combine(directory, "ctc-fp32.onemodel");
            custom = new([new() { Path = graph, Channels = 9812, Stride = 320, ReceptiveField = 400 }], GraphFrontEnd.RawWaveform, vocabulary, 0, TimeSpan.FromMilliseconds(20), info, threads: 4);
            ctc = Session.Load(graph, new ModelOptions { IntraOpThreads = 4, MaxCachedPlans = 1 });
        }
        else
        {
            string[] pieces = new string[8193]; pieces[8192] = "";
            using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "tokenizer.json")));
            foreach (var piece in document.RootElement.GetProperty("model").GetProperty("vocab").EnumerateObject())
                pieces[piece.Value.GetInt32()] = piece.Name == "<unk>" ? " ⁇ " : piece.Name;
            vocabulary = new(pieces, metaspace: true, excludeTrailingPunctuationFromTiming: true);
            byte[] bytes = File.ReadAllBytes(Path.Combine(directory, "embedding.f32")); var embeddings = new float[bytes.Length / 4];
            for (int i = 0; i < embeddings.Length; i++) embeddings[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
            custom = new([
                new() { Path = Path.Combine(directory, "encoder-part0.onemodel"), Channels = 1024, Stride = 8, SamePadding = true },
                new() { Path = Path.Combine(directory, "encoder-part1.onemodel"), Channels = 1024 }
            ], GraphFrontEnd.Nemo128, vocabulary, 8192, TimeSpan.FromMilliseconds(80), info,
                new() { PredictorPath = Path.Combine(directory, "predictor.onemodel"), JointPath = Path.Combine(directory, "joint.onemodel"), Embeddings = embeddings, Durations = [0, 1, 2, 3, 4] }, threads: 4);
            tdt = ParakeetRecognizer.Load(new(directory, info), threads: 4);
        }
        using (custom) using (ctc) using (tdt)
        {
            int checkedFiles = 0;
            foreach (string path in Directory.EnumerateFiles(corpus, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(20))
            {
                string id = Path.GetFileNameWithoutExtension(path), target = Path.Combine(output, id + ".json");
                var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
                var audio = new AudioBuffer(clip.Samples, 16000);
                var actual = custom.Transcribe(audio, "en"); string expected;
                if (tdt is not null)
                {
                    var builtIn = tdt.Transcribe(audio, "en"); expected = builtIn.Text;
                    if (!actual.Words.SequenceEqual(builtIn.Words)) throw new InvalidDataException("TDT word/timing mismatch: " + id);
                }
                else
                {
                    // Independent family probe path: frame argmax and normal-character filtering.
                    using var input = Tensor.FromSpan<float>(audio.Samples.Span, Shape.Create(1, 1, audio.Samples.Length));
                    using var result = ctc!.Run(new Dictionary<string, Tensor> { ["waveform"] = input });
                    var logits = result["logits"].Data<float>(); var ids = new List<int>(); int previous = -1;
                    for (int frame = 0; frame < logits.Length / 9812; frame++)
                    {
                        int best = 0;
                        for (int token = 1; token < 9812; token++) if (logits[frame * 9812 + token] > logits[frame * 9812 + best]) best = token;
                        if (best != previous && best != 0) ids.Add(best);
                        previous = best;
                    }
                    expected = string.Concat(ids.Select(i => vocabulary.Pieces[i]));
                }
                if (actual.Text != expected) throw new InvalidDataException("Text mismatch: " + id);
                File.WriteAllText(target, actual.ToJson()); checkedFiles++; Console.WriteLine(kind + " " + id + " identical");
            }
            if (checkedFiles != 20) throw new InvalidDataException("Expected 20 clips.");
            File.WriteAllText(Path.Combine(output, "summary.txt"), kind + " 20/20 identical; TDT includes exact word/timing/confidence equality.\n");
        }
    }
    private sealed class NoWeights : OneSpeech.Formats.IFloatTensorSource
    {
        public long[] GetShape(string name) => throw new InvalidOperationException("Synthetic fixture has no learned weights.");
        public void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken) => throw new InvalidOperationException("Synthetic fixture has no learned weights.");
    }
}
