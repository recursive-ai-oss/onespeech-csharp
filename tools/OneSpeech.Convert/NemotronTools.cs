using System.Buffers.Binary;
using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Diarization;

internal static class NemotronTools
{
    internal static void TopKProbe(string directory)
    {
        foreach (string file in Directory.EnumerateFiles(directory, "*.scores").Order(StringComparer.Ordinal))
        {
            string name = Path.GetFileNameWithoutExtension(file);
            int count = int.Parse(name.Split('-')[1], System.Globalization.CultureInfo.InvariantCulture);
            float[] scores = Read(file);
            byte[] data = File.ReadAllBytes(Path.ChangeExtension(file, ".ids"));
            int[] expected = new int[data.Length / 8];
            for (int i = 0; i < expected.Length; i++) expected[i] = checked((int)BinaryPrimitives.ReadInt64LittleEndian(data.AsSpan(i * 8)));
            Array.Sort(expected);
            foreach (bool previous in new[] { true, false })
            {
                int[] actual = CacheTopScores.Select(scores, count, previous); Array.Sort(actual);
                Console.WriteLine($"{name} nth={(previous ? "K-1" : "K")} matches={actual.Intersect(expected).Count()}/{expected.Length}");
            }
        }
    }
    internal static void CacheMap(string native, string reference, string silencePath)
    {
        float[] frames = Read(Path.Combine(native, "embedded0.f32")), silence = Read(silencePath);
        float[] ours = Read(Path.Combine(native, "cache0.f32")), theirs = Read(Path.Combine(reference, "cache0.f32"));
        using var writer = new StreamWriter(Path.Combine(native, "cache-map.tsv"));
        writer.WriteLine("slot\tnative_frame\treference_frame\tnative_error\treference_error");
        int equal = 0;
        int[] referenceIds = new int[304];
        double worst = 0;
        for (int slot = 0; slot < 304; slot++)
        {
            var a = Match(ours.AsSpan(slot * 512, 512)); var b = Match(theirs.AsSpan(slot * 512, 512));
            writer.WriteLine($"{slot}\t{a.Frame}\t{b.Frame}\t{a.Error:R}\t{b.Error:R}");
            referenceIds[slot] = b.Frame;
            if (a.Frame == b.Frame) equal++;
            worst = Math.Max(worst, b.Error);
        }
        Console.WriteLine($"cache-map-identical-slots={equal}/304 worst-nearest-L2={worst:R}");
        var probabilities = Read(Path.Combine(native, "full-prob0.f32"));
        {
            var cache = new ArrivalOrderSpeakerCache(silence); cache.Update(frames, probabilities, 340);
            var candidate = cache.Window([]); int matching = 0;
            for (int t = 0; t < 304; t++)
            {
                double error = 0;
                for (int d = 0; d < 512; d++) error = Math.Max(error, Math.Abs(candidate[t * 512 + d] - theirs[t * 512 + d]));
                if (error < .001) matching++;
            }
            var ids = new Dictionary<float, int>(); for (int t = 0; t < 380; t++) ids.TryAdd(frames[t * 512], t); ids[silence[0]] = -1;
            int[] selectedIds = Enumerable.Range(0, 264).Select(t => ids[candidate[t * 512]]).Where(t => t >= 0).ToArray();
            int[] selectedExpected = referenceIds.Take(264).Where(t => t >= 0).ToArray();
            int filtered = selectedIds.Zip(selectedExpected).Count(p => p.First == p.Second);
            Console.WriteLine($"mean-pooling matching-cache-rows={matching}/304 positive-rows={selectedIds.Length}/{selectedExpected.Length} filtered-match={filtered}");
        }
        (int Frame, double Error) Match(ReadOnlySpan<float> frame)
        {
            int best = -1; double bestError = double.PositiveInfinity;
            for (int t = 0; t <= 380; t++)
            {
                ReadOnlySpan<float> candidate = t == 380 ? silence : frames.AsSpan(t * 512, 512);
                double error = 0;
                for (int d = 0; d < 512; d++) { double delta = frame[d] - candidate[d]; error += delta * delta; }
                if (error < bestError) { best = t == 380 ? -1 : t; bestError = error; }
            }
            return (best, bestError);
        }
    }
    internal static void StreamProbe(string directory, string audio, string output, string reference)
    {
        Directory.CreateDirectory(output);
        using var source = AudioFileSource.Open(audio, CancellationToken.None);
        using var session = new NemotronDiarizationSession(directory, 4, CancellationToken.None);
        var cache = new ArrivalOrderSpeakerCache(Read(Path.Combine(directory, "silence.f32")));
        int step = 0;
        foreach (var chunk in NemotronFeatureChunks.Read(VadChunker.Blocks(source, CancellationToken.None), CancellationToken.None).Take(3))
        {
            int prefix = cache.CachedFrames + cache.FifoFrames;
            var embedded = session.Embed(chunk.Stacked, CancellationToken.None);
            Write(Path.Combine(output, "embedded" + step + ".f32"), embedded);
            var window = cache.Window(embedded);
            var result = session.Forward(window, CancellationToken.None);
            string f = Path.Combine(output, "features" + step + ".f32"); Write(f, chunk.Stacked);
            Console.Write("features" + step + " "); Compare(f, Path.Combine(reference, "features" + step + ".f32"));
            f = Path.Combine(output, "prob" + step + ".f32"); Write(f, result.Probabilities.AsSpan(prefix * 64, chunk.OutputMelFrames * 8));
            Console.Write("prob" + step + " "); Compare(f, Path.Combine(reference, "prob" + step + ".f32"));
            Write(Path.Combine(output, "full-prob" + step + ".f32"), result.Probabilities);
            cache.Update(window, result.Probabilities, Math.Min(340, chunk.Frames));
            f = Path.Combine(output, "cache" + step + ".f32"); Write(f, cache.Window([]));
            Console.Write("cache" + step + " "); Compare(f, Path.Combine(reference, "cache" + step + ".f32"));
            step++;
        }
        CacheMap(output, reference, Path.Combine(directory, "silence.f32"));
    }
    internal static void CacheProbe(string directory)
    {
        foreach (string name in new[] { "default", "quota-2", "quota-4", "quota-8" })
        {
            string probabilitiesPath = name == "default" ? "scoring-probs.f32" : name + ".probs";
            string selectedPath = name == "default" ? "default-selected.f32" : name + ".selected";
            if (!File.Exists(Path.Combine(directory, probabilitiesPath))) continue;
            float[] means = Read(Path.Combine(directory, probabilitiesPath));
            float[] expected = Read(Path.Combine(directory, selectedPath));
            float[] window = new float[340 * 512], probabilities = new float[340 * 64];
            for (int t = 0; t < 340; t++)
            {
                Array.Fill(window, t + 1, t * 512, 512);
                for (int k = 0; k < 8; k++) means.AsSpan(t * 8, 8).CopyTo(probabilities.AsSpan((t * 8 + k) * 8, 8));
            }
            var cache = new ArrivalOrderSpeakerCache(new float[512]); cache.Update(window, probabilities, 340);
            var actual = cache.Window([]);
            int matches = 0;
            for (int t = 0; t < expected.Length; t++) if (actual[t * 512] == expected[t]) matches++;
            Console.WriteLine($"{name}_cache_matches={matches}/{expected.Length}");
            Write(Path.Combine(directory, "native-" + name + ".selected"), Enumerable.Range(0, expected.Length).Select(t => actual[t * 512]).ToArray());
            if (matches != expected.Length) throw new InvalidDataException("Public speaker-cache fixture mismatch: " + name);
        }
    }
    internal static void Features(string audio, string output)
    {
        var clip = OneAudio.Audio.Load(audio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        var chunk = NemotronFeatureChunks.Read([new AudioBuffer(clip.Samples, 16000)], CancellationToken.None).First();
        Write(output, chunk.Stacked.AsSpan(0, chunk.OutputMelFrames * 128));
    }
    internal static void Convert(string input, string output) => NemotronDiarizationConverter.Convert(input, output, CancellationToken.None);
    internal static void Probe(string directory, string featuresPath, string output)
    {
        float[] features = Read(featuresPath);
        int melFrames = features.Length / 128, frames = (melFrames + 7) / 8;
        float[] stacked = new float[frames * 1024]; features.CopyTo(stacked, 0);
        using var session = new NemotronDiarizationSession(directory, 4, CancellationToken.None);
        var embedded = session.Embed(stacked, CancellationToken.None);
        var (probabilities, encoded, upsampled) = session.Forward(embedded, CancellationToken.None, diagnostics: true);
        Directory.CreateDirectory(output);
        Write(Path.Combine(output, "probabilities.f32"), probabilities);
        Write(Path.Combine(output, "encoded.f32"), encoded);
        Write(Path.Combine(output, "upsampled.f32"), upsampled);
        Console.WriteLine($"mel_frames={melFrames} encoder_frames={frames}");
    }
    internal static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return values;
    }
    internal static void Write(string path, ReadOnlySpan<float> values)
    {
        using var file = File.Create(path);
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        file.Write(bytes);
    }
    internal static void Compare(string first, string second)
    {
        float[] a = Read(first), b = Read(second);
        int length = Math.Min(a.Length, b.Length);
        double sum = 0, max = 0;
        for (int i = 0; i < length; i++) { double error = Math.Abs(a[i] - b[i]); sum += error; max = Math.Max(max, error); }
        Console.WriteLine($"length={a.Length}/{b.Length} max={max:R} mean={sum / length:R}");
    }
}
