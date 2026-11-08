using OneSpeech.Text;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class OmniProbe
{
    internal static void RunBatch(string graph, string tokenizer, string directory, string destination, int count)
    {
        var vocabulary = SentencePieceProto.ReadFile(tokenizer);
        using var session = Session.Load(graph, new ModelOptions { IntraOpThreads = 4, MaxCachedPlans = 1 });
        using var file = File.Create(destination);
        foreach (string audio in Directory.EnumerateFiles(directory, "*.flac", SearchOption.AllDirectories).OrderBy(p => Path.GetFileNameWithoutExtension(p), StringComparer.Ordinal).Take(count))
        {
            var clip = OneAudio.Audio.Load(audio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            using var input = Tensor.FromSpan<float>(clip.Samples.Span, Shape.Create(1, 1, clip.Samples.Length));
            using var output = session.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["waveform"] = input });
            var logits = output["logits"].Data<float>();
            List<int> ids = [];
            int previous = -1;
            for (int frame = 0; frame < logits.Length / vocabulary.Pieces.Length; frame++)
            {
                int best = 0;
                for (int token = 1; token < vocabulary.Pieces.Length; token++)
                    if (logits[frame * vocabulary.Pieces.Length + token] > logits[frame * vocabulary.Pieces.Length + best]) best = token;
                if (best != previous && vocabulary.Types[best] == 1) ids.Add(best);
                previous = best;
            }
            using (var writer = new System.Text.Json.Utf8JsonWriter(file))
            {
                writer.WriteStartObject(); writer.WriteString("id", Path.GetFileNameWithoutExtension(audio));
                writer.WriteString("text", string.Concat(ids.Select(i => vocabulary.Pieces[i])));
                writer.WriteEndObject(); writer.Flush();
            }
            file.WriteByte(10); file.Flush(); Console.WriteLine(Path.GetFileNameWithoutExtension(audio));
        }
    }

    internal static void Run(string graph, string tokenizer, string audio)
    {
        var vocabulary = SentencePieceProto.ReadFile(tokenizer);
        Console.WriteLine($"pieces={vocabulary.Pieces.Length}; multi-character-normal-pieces={vocabulary.Pieces.Where((p, i) => vocabulary.Types[i] == 1 && p.EnumerateRunes().Count() != 1).Count()}");
        using var session = Session.Load(graph, new ModelOptions { IntraOpThreads = 4, MaxCachedPlans = 1 });
        var clip = OneAudio.Audio.Load(audio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        using var input = Tensor.FromSpan<float>(clip.Samples.Span, Shape.Create(1, 1, clip.Samples.Length));
        using var output = session.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["waveform"] = input });
        float[] logits = output["logits"].Data<float>().ToArray();
        int frames = logits.Length / vocabulary.Pieces.Length;
        List<int> ids = [];
        int previous = -1;
        for (int frame = 0; frame < frames; frame++)
        {
            int best = 0;
            for (int token = 1; token < vocabulary.Pieces.Length; token++)
                if (logits[frame * vocabulary.Pieces.Length + token] > logits[frame * vocabulary.Pieces.Length + best]) best = token;
            if (best != previous) ids.Add(best);
            previous = best;
        }
        Console.WriteLine($"frames={frames}; ids={string.Join(',', ids)}");
        Console.WriteLine(string.Concat(ids.Where(i => vocabulary.Types[i] == 1).Select(i => vocabulary.Pieces[i])));
    }
}
