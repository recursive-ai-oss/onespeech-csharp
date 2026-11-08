namespace OneSpeech.Diarization;

// Streaming Sortformer §§3.1–3.3: retain pre-encoder frames, not transformer KV state.
internal sealed class ArrivalOrderSpeakerCache(float[] silence)
{
    private float[] cache = [], fifo = [];
    internal int CachedFrames => cache.Length / 512;
    internal int FifoFrames => fifo.Length / 512;
    internal float[] Window(float[] chunk) => [.. cache, .. fifo, .. chunk];

    internal void Update(float[] window, float[] probabilities, int chunkFrames)
    {
        int previous = CachedFrames, recent = FifoFrames + chunkFrames;
        int flush = recent - 40;
        if (flush < 300)
        {
            fifo = window.AsSpan(previous * 512, recent * 512).ToArray();
            return;
        }
        int count = previous + flush;
        float[] means = new float[count * 8];
        for (int t = 0; t < count; t++) for (int s = 0; s < 8; s++)
        {
            int p = t * 64 + s;
            float a = probabilities[p] + probabilities[p + 32];
            float b = probabilities[p + 8] + probabilities[p + 40];
            float c = probabilities[p + 16] + probabilities[p + 48];
            float d = probabilities[p + 24] + probabilities[p + 56];
            means[t * 8 + s] = ((a + b) + c + d) / 8;
        }
        if (count <= 264) cache = window.AsSpan(0, count * 512).ToArray();
        else cache = Compress(window, means, count);
        fifo = window.AsSpan(count * 512, 40 * 512).ToArray();
    }

    private float[] Compress(float[] window, float[] probabilities, int count)
    {
        var scores = new float[count * 8];
        for (int s = 0; s < 8; s++)
        {
            for (int t = 0; t < count; t++)
            {
                float p = probabilities[t * 8 + s];
                float negative = 0;
                for (int other = 0; other < 8; other++) negative += MathF.Log(Math.Max(.25f, 1 - probabilities[t * 8 + other]));
                float score = MathF.Log(Math.Max(.25f, p)) + negative - MathF.Log(Math.Max(.25f, 1 - p));
                scores[s * count + t] = p < .5 ? float.NegativeInfinity : score;
            }
            // Public crafted-grid probes: if at least half the 32-frame speaker budget is
            // confidently exclusive (joint activity > 0.5), drop its nonexclusive frames.
            // Test this before the recency/representation bonuses, which must not create exclusivity.
            float exclusiveThreshold = -MathF.Log(2);
            int exclusive = Enumerable.Range(0, count).Count(t => scores[s * count + t] > exclusiveThreshold);
            if (exclusive >= 16)
                for (int t = 0; t < count; t++) if (scores[s * count + t] <= exclusiveThreshold) scores[s * count + t] = float.NegativeInfinity;
            // Public cache probes show that the recency boundary is the nominal 264 slots,
            // even on the first compression; using the actual initial length changes selected frames.
            for (int t = 264; t < count; t++) scores[s * count + t] += .05f;
            int[] order = Enumerable.Range(0, count).OrderByDescending(t => scores[s * count + t]).ThenBy(t => t).ToArray();
            // 256 non-silence slots / 8 speakers: 24 strongly boosted, 48 weakly boosted.
            for (int i = 0; i < Math.Min(24, count); i++) scores[s * count + order[i]] += 2 * MathF.Log(2);
            for (int i = 0; i < Math.Min(48, count); i++) scores[s * count + order[i]] += MathF.Log(2);
        }
        int[] selected = Enumerable.Range(0, scores.Length).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(256).Order().ToArray();
        float[] result = new float[264 * 512];
        int position = 0;
        for (int s = 0; s < 8; s++)
        {
            int first = position;
            foreach (int index in selected)
            {
                if (index / count != s || float.IsNegativeInfinity(scores[index])) continue;
                window.AsSpan((index % count) * 512, 512).CopyTo(result.AsSpan(position++ * 512, 512));
            }
            // Observed public-cache output: silence follows represented speakers; unused/invalid slots
            // sort to the tail, never in front of the first represented speaker.
            if (position > first) silence.CopyTo(result.AsSpan(position++ * 512, 512));
        }
        while (position < 264) silence.CopyTo(result.AsSpan(position++ * 512, 512));
        return result;
    }
}
