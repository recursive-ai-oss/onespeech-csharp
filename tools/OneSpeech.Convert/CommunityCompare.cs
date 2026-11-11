using System.Buffers.Binary;
using OneSpeech;
using OneSpeech.Diarization;
using OneSpeech.Formats;

// Compares the native community-pipeline stages with hook artifacts recorded from the released reference.
internal static class CommunityCompare
{
    internal static float[] ReadWav(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        int pos = 12, format = 0, bits = 0, channels = 1;
        while (pos + 8 <= b.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(b, pos, 4);
            int size = BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(pos + 4));
            if (id == "fmt ") { format = BinaryPrimitives.ReadUInt16LittleEndian(b.AsSpan(pos + 8)); channels = BinaryPrimitives.ReadUInt16LittleEndian(b.AsSpan(pos + 10)); bits = BinaryPrimitives.ReadUInt16LittleEndian(b.AsSpan(pos + 22)); }
            if (id == "data")
            {
                int n = Math.Min(size, b.Length - pos - 8) / (bits / 8) / channels;
                var r = new float[n];
                for (int i = 0; i < n; i++)
                    r[i] = bits == 32 ? BinaryPrimitives.ReadSingleLittleEndian(b.AsSpan(pos + 8 + i * channels * 4)) : BinaryPrimitives.ReadInt16LittleEndian(b.AsSpan(pos + 8 + i * channels * 2)) / 32768f;
                return r;
            }
            pos += 8 + size + (size & 1);
        }
        throw new InvalidDataException("no data chunk");
    }

    internal static string Find(string dir, string suffix) => Path.GetFileName(Directory.GetFiles(dir, "*" + suffix).Single())!;
    private static NpyArray Load(string dir, string name) => NpzArchive.ParseNpy(File.ReadAllBytes(Path.Combine(dir, name)));

    internal static void Run(string snapshot, string cache, string audioPath, string dir)
    {
        float[] audio = ReadWav(audioPath);
        var seg = Load(dir, Find(dir, "-segmentation.npy"));
        var emb = Load(dir, Directory.GetFiles(dir, "*-embeddings.npy").Select(Path.GetFileName).Order(StringComparer.Ordinal).Last()!);
        int chunks = seg.Shape[0];
        Console.WriteLine($"audio samples={audio.Length} ({audio.Length / 16000.0:F3} s) ref chunks={chunks} emb shape={string.Join("x", emb.Shape)}");

        using var segmenter = PyannoteSegmenter.LoadCheckpoint(Path.Combine(snapshot, "segmentation/pytorch_model.bin"), snapshot, "pyannote/speaker-diarization-community-1", "pyannote-community-1-segmentation", null, cache, 4, default);
        string graphPath = Path.Combine(cache, "wespeaker-frames.onemodel");
        WeSpeakerResNetConverter converter;
        using (var input = File.OpenRead(Path.Combine(snapshot, "embedding/pytorch_model.bin"))) converter = new WeSpeakerResNetConverter(TensorCheckpoint.Read(input, segmentation: true));
        converter.Save(graphPath);
        var (w, bias) = converter.Projection();
        using var embedder = new WeSpeakerEmbedder(graphPath, w, bias, 4, default);

        int segMismatchFrames = 0, segFrames = 0;
        var cosines = new List<double>(); int nanRef = 0;
        var cleanOfNan = new List<int>(); var cleanOfValid = new List<int>();
        for (int c = 0; c < chunks; c++)
        {
            var chunk = new float[160000];
            int start = c * 16000, n = Math.Max(0, Math.Min(160000, audio.Length - start));
            Array.Copy(audio, start, chunk, 0, n);
            float[] lp = segmenter.GetFrameLogProbabilities(new AudioBuffer(chunk, 16000), default);
            var ours = new int[589];
            for (int f = 0; f < 589; f++)
            {
                int best = 0; for (int k = 1; k < 7; k++) if (lp[f * 7 + k] > lp[f * 7 + best]) best = k;
                ours[f] = PyannoteSegmenter.ClassMasks[best];
                int refMask = 0; for (int k = 0; k < 3; k++) if (seg.Values[(c * 589 + f) * 3 + k] > 0.5) refMask |= 1 << k;
                segFrames++; if (refMask != ours[f]) segMismatchFrames++;
            }
            float[] frames = embedder.Frames(chunk, out int steps, default);
            for (int k = 0; k < 3; k++)
            {
                var weights = new float[589]; int clean = 0, active = 0;
                for (int f = 0; f < 589; f++)
                {
                    int m = (int)seg.Values[(c * 589 + f) * 3 + 0] | (int)seg.Values[(c * 589 + f) * 3 + 1] << 1 | (int)seg.Values[(c * 589 + f) * 3 + 2] << 2;
                    bool on = (m >> k & 1) != 0; if (on) active++;
                    if (on && (m & (m - 1)) == 0) { weights[f] = 1; clean++; }
                }
                bool refNan = double.IsNaN(emb.Values[(c * 3 + k) * 256]);
                if (refNan) { nanRef++; cleanOfNan.Add(clean); }
                else cleanOfValid.Add(clean);
                double refNorm = 0; for (int i = 0; i < 256; i++) refNorm += emb.Values[(c * 3 + k) * 256 + i] * emb.Values[(c * 3 + k) * 256 + i];
                if (clean < 40 && (active > 0 || clean > 0 || c < 3)) Console.WriteLine($"row c={c} k={k} active={active} clean={clean} refNorm={Math.Sqrt(refNorm):F3}");
                if (clean == 0 || refNan) continue;
                float[] ours256 = embedder.Project(WeSpeakerEmbedder.Pool(frames, steps, weights));
                double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < 256; i++) { double a = ours256[i], r = emb.Values[(c * 3 + k) * 256 + i]; dot += a * r; na += a * a; nb += r * r; }
                cosines.Add(dot / Math.Sqrt(na * nb));
            }
        }
        Console.WriteLine($"segmentation: {segMismatchFrames}/{segFrames} frames differ from the reference");
        cosines.Sort();
        Console.WriteLine($"embeddings compared={cosines.Count} cosine min={cosines[0]:F8} p10={cosines[cosines.Count / 10]:F8} median={cosines[cosines.Count / 2]:F8}; reference NaN rows={nanRef}");
        Console.WriteLine($"clean frames of NaN rows: max={(cleanOfNan.Count > 0 ? cleanOfNan.Max() : -1)}; of valid rows: min={cleanOfValid.Min()}");
    }
}
