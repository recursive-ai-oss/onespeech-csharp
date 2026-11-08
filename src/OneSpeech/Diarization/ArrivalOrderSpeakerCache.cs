namespace OneSpeech.Diarization;

// Streaming Sortformer §§3.1–3.3: retain pre-encoder frames, not transformer KV state.
internal sealed class ArrivalOrderSpeakerCache(float[] silence)
{
    private float[] cache = [], fifo = [], cacheMeans = [];
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
        // Cached frames keep the activity that selected them, rather than being relabelled by
        // later chunks (observed using a two-speaker public cache probe).
        cacheMeans.CopyTo(means, 0);
        if (count <= 264)
        {
            cache = window.AsSpan(0, count * 512).ToArray();
            cacheMeans = means;
        }
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
            float[] speakerScores = scores.AsSpan(s * count, count).ToArray();
            int[] strong = CacheTopScores.Select(speakerScores, Math.Min(24, count));
            // 256 non-silence slots / 8 speakers: 24 strongly boosted, 48 weakly boosted.
            foreach (int index in strong) speakerScores[index] += 2 * MathF.Log(2);
            int[] weak = CacheTopScores.Select(speakerScores, Math.Min(48, count));
            foreach (int index in weak) speakerScores[index] += MathF.Log(2);
            speakerScores.CopyTo(scores.AsSpan(s * count, count));
        }
        // Include the learned-silence column in the partition: it participates in tie selection.
        int width = count + 1;
        float[] withSilence = new float[width * 8];
        for (int s = 0; s < 8; s++)
        {
            scores.AsSpan(s * count, count).CopyTo(withSilence.AsSpan(s * width, count));
            withSilence[s * width + count] = float.PositiveInfinity;
        }
        int[] selected = CacheTopScores.Select(withSilence, 264).Order().ToArray();
        float[] result = new float[264 * 512];
        float[] retainedMeans = new float[264 * 8];
        int position = 0;
        for (int s = 0; s < 8; s++)
        {
            int first = position;
            foreach (int index in selected)
            {
                if (index / width != s || index % width == count || float.IsNegativeInfinity(withSilence[index])) continue;
                probabilities.AsSpan((index % width) * 8, 8).CopyTo(retainedMeans.AsSpan(position * 8, 8));
                window.AsSpan((index % width) * 512, 512).CopyTo(result.AsSpan(position++ * 512, 512));
            }
            // Observed public-cache output: silence follows represented speakers; unused/invalid slots
            // sort to the tail, never in front of the first represented speaker.
            if (position > first) silence.CopyTo(result.AsSpan(position++ * 512, 512));
        }
        while (position < 264) silence.CopyTo(result.AsSpan(position++ * 512, 512));
        cacheMeans = retainedMeans;
        return result;
    }
}
