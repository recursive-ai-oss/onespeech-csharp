using System.Diagnostics;
using OneSpeech;
using OneSpeech.Qwen3;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

// Diagnostics stay outside the library: timings, process memory and the public runtime pool counters.
internal static class QwenMemoryProbe
{
    internal static int Synthetic(string directory)
    {
        Directory.CreateDirectory(directory);
        const int width = 1024;
        string manifest = "{\"format\":\"onemodel\",\"version\":1,\"ir_version\":8,\"opsets\":[{\"domain\":\"\",\"version\":17}],\"graph\":{\"name\":\"linear-memory\",\"inputs\":[{\"name\":\"x\",\"type\":{\"kind\":\"tensor\",\"elem\":1,\"shape\":[1,1024]}}],\"outputs\":[{\"name\":\"y\",\"type\":{\"kind\":\"tensor\",\"elem\":1,\"shape\":[1,1024]}}],\"initializers\":[{\"name\":\"w\",\"dtype\":1,\"dims\":[1024,1024],\"data\":{\"offset\":0,\"length\":4194304}}],\"nodes\":[{\"op\":\"Transpose\",\"inputs\":[\"w\"],\"outputs\":[\"wt\"],\"attributes\":[{\"name\":\"perm\",\"type\":7,\"value\":[1,0]}]},{\"op\":\"MatMul\",\"inputs\":[\"x\",\"wt\"],\"outputs\":[\"y\"],\"attributes\":[]}]}}";
        byte[] json = System.Text.Encoding.UTF8.GetBytes(manifest), header = new byte[64];
        "ONEMODEL"u8.CopyTo(header);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), 1);
        long offset = (64 + json.Length + 63) / 64 * 64;
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(16), 64);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(24), json.Length);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(32), offset);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(40), width * width * 4);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(header.AsSpan(48), offset + width * width * 4);
        using (var stream = File.Create(Path.Combine(directory, "head.onemodel")))
        using (var writer = new BinaryWriter(stream))
        {
            writer.Write(header); writer.Write(json); stream.Position = offset;
            for (int i = 0; i < width * width; i++) writer.Write((i % 251 - 125) / 128f);
        }
        return LoadGraphs(directory);
    }

    internal static int LoadGraphs(string directory)
    {
        var sessions = new List<Session>();
        try
        {
            Snapshot("start", 0);
            long bytes = 0;
            foreach (string file in Directory.GetFiles(directory, "*.onemodel").Where(p => !p.EndsWith("-bf16.onemodel", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
            {
                sessions.Add(Session.Load(file, new ModelOptions { IntraOpThreads = 8, MaxCachedPlans = 1 }));
                bytes += new FileInfo(file).Length;
                Snapshot(Path.GetFileName(file), bytes);
            }
            // Distinguish unreachable managed load temporaries and idle pooled blocks from live graph storage.
            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); BufferPool.Shared.Trim();
            Snapshot("trimmed", bytes);
        }
        finally { foreach (var session in sessions) session.Dispose(); }
        Snapshot("disposed", 0);
        return 0;
    }

    internal static int Slice(string directory, string corpus, string output)
    {
        Directory.CreateDirectory(output);
        Snapshot("start", 0);
        using var recognizer = Qwen3Recognizer.Load(directory, new Qwen3Options { Threads = 8 });
        Snapshot("loaded", 0);
        foreach (string path in Directory.EnumerateFiles(corpus, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileName, StringComparer.Ordinal).Take(20))
        {
            var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var transcript = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
            File.WriteAllText(Path.Combine(output, Path.GetFileNameWithoutExtension(path) + ".json"), transcript.ToJson());
            Snapshot(Path.GetFileName(path), 0);
        }
        return 0;
    }

    internal static int Compare(string before, string after)
    {
        int count = 0;
        foreach (string file in Directory.GetFiles(after, "*.json").Where(p => !p.EndsWith(".measurement.json", StringComparison.Ordinal)))
        {
            var old = Transcript.FromJson(File.ReadAllText(Path.Combine(before, Path.GetFileName(file))));
            var current = Transcript.FromJson(File.ReadAllText(file));
            if (old.Text != current.Text || !old.Words.Select(w => w.Text).SequenceEqual(current.Words.Select(w => w.Text)))
                throw new InvalidDataException("Recognition changed: " + Path.GetFileName(file));
            count++;
        }
        if (count == 0) throw new InvalidDataException("No hypotheses to compare.");
        Console.WriteLine($"Identical recognized text and written words: {count} files.");
        return 0;
    }

    private static void Snapshot(string phase, long fileBytes)
    {
        using var process = Process.GetCurrentProcess();
        var pool = BufferPool.Shared.GetStats();
        Console.WriteLine(FormattableString.Invariant($"{phase}\tfiles={fileBytes}\trss={process.WorkingSet64}\tpeak={process.PeakWorkingSet64}\theap={GC.GetTotalMemory(false)}\tpoolLive={pool.OutstandingBytes}\tpoolIdle={pool.IdleBytes}"));
    }
}
