using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>Overlap-aware offline diarization using the explicitly loaded segmentation-3.0 model and a borrowed 16 kHz embedder. Uses our clustering, not the community-1 pipeline. Models are borrowed and must outlive processing; embeddings are never persisted.</summary>
public sealed class PyannoteDiarizer : BlockDiarizer
{
    private sealed record Chunk(long Offset, int Length, int[] Activity, int[] Windows);
    private readonly PyannoteSegmenter segmentation;
    private readonly SpeakerEmbedder embedder;
    private readonly ClusteringOptions options;

    /// <summary>Creates a diarizer with independent 10 s segmentation windows, exclusive-speech embeddings, and existing spectral/AHC clustering. Options default to ClusteringOptions defaults; resegmentation, WindowSeconds and ShiftSeconds do not apply to local-speaker masks. A chunk's local speakers are assigned distinct global clusters. Throws ArgumentException for invalid options or embedder dimensions/rate. No resources are owned.</summary>
    public PyannoteDiarizer(PyannoteSegmenter segmentation, SpeakerEmbedder embedder, ClusteringOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(segmentation); ArgumentNullException.ThrowIfNull(embedder);
        if (embedder.SampleRate != 16000 || embedder.Dimension is < 1 or > 4096) throw new ArgumentException("A 16 kHz embedder of dimension 1–4096 is required.", nameof(embedder));
        this.segmentation = segmentation; this.embedder = embedder;
        this.options = options ?? new ClusteringOptions(); this.options.Validate();
    }

    /// <summary>Maximum configured global speaker count. At most two simultaneous speakers are detected in each frame; live streaming is not supported.</summary>
    public override Capabilities Capabilities => new() { MaxSpeakers = options.MaxSpeakers };
    /// <inheritdoc />
    public override ImmutableArray<ModelInfo> Models => [segmentation.Model, embedder.Model];

    /// <summary>Processes consecutive 16 kHz blocks, downmixing up to 32 channels. Retains bounded 30 s PCM plus O(ND) embeddings and O(N) local activity masks (N ≤ MaxWindows, D ≤4096). Short final windows are padded and clipped. Fewer than 100 ms of usable speaker audio yields no embedding/turn for that local speaker. Empty audio returns no turns. Throws CapabilityNotSupportedException if the window bound is exceeded or a count hint cannot represent the local speakers; checks cancellation per chunk, embedding and clustering sweep.</summary>
    public override ImmutableArray<Turn> DiarizeBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        var windows = new List<EmbeddingWindow>();
        var chunks = new List<Chunk>();
        long offset = 0;
        int requiredSpeakers = 1;
        foreach (var block in DiarizationBlocks.Fixed(blocks, cancellationToken))
        {
            for (int local = 0; local < block.FrameCount; local += PyannoteSegmenter.Samples)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int length = Math.Min(PyannoteSegmenter.Samples, block.FrameCount - local);
                float[] samples = new float[PyannoteSegmenter.Samples];
                for (int i = 0; i < length; i++)
                {
                    double value = 0;
                    for (int c = 0; c < block.Channels; c++) value += block.Samples.Span[(local + i) * block.Channels + c];
                    samples[i] = (float)(value / block.Channels);
                }
                int[] activity = segmentation.Activity(new AudioBuffer(samples, 16000), cancellationToken);
                int[] indices = [-1, -1, -1];
                for (int speaker = 0; speaker < 3; speaker++)
                {
                    float[] selected = Select(samples, length, activity, speaker, exclusive: true);
                    if (selected.Length < 1600) selected = Select(samples, length, activity, speaker, exclusive: false);
                    if (selected.Length < 1600) continue;
                    if (windows.Count >= options.MaxWindows) throw new CapabilityNotSupportedException("Segmentation diarization exceeded ClusteringOptions.MaxWindows.");
                    float[] vector = new float[embedder.Dimension];
                    embedder.Embed(new AudioBuffer(selected, 16000), vector, cancellationToken);
                    EmbeddingClustering.Normalize(vector);
                    indices[speaker] = windows.Count;
                    windows.Add(new EmbeddingWindow { Start = TimeSpan.FromSeconds((offset + local) / 16000.0), End = TimeSpan.FromSeconds((offset + local + length) / 16000.0), Embedding = vector });
                }
                requiredSpeakers = Math.Max(requiredSpeakers, indices.Count(i => i >= 0));
                chunks.Add(new Chunk(offset + local, length, activity, indices));
                if (chunks.Count > options.MaxWindows) throw new CapabilityNotSupportedException("Segmentation diarization exceeded its retained chunk bound.");
            }
            offset += block.FrameCount;
        }
        if (windows.Count == 0) return [];
        if (requiredSpeakers > options.MaxSpeakers || options.NumSpeakers is { } count && count < requiredSpeakers)
            throw new CapabilityNotSupportedException("The requested speaker count is smaller than the segmentation model's local speaker count.");
        var clustering = options with { MinSpeakers = Math.Max(options.MinSpeakers, requiredSpeakers) };
        int[] labels = EmbeddingClustering.Cluster(windows, clustering, cancellationToken);
        int k = labels.Max() + 1;
        var centers = new float[k][];
        for (int i = 0; i < k; i++) centers[i] = new float[embedder.Dimension];
        for (int i = 0; i < windows.Count; i++)
            for (int d = 0; d < embedder.Dimension; d++) centers[labels[i]][d] += windows[i].Embedding.Span[d];
        foreach (var center in centers)
        {
            double norm = 0; foreach (float value in center) norm += (double)value * value;
            // Opposite valid embeddings can cancel. A zero centroid supplies no cosine
            // evidence; retain it so stable assignment ties still preserve count hints.
            if (norm > 0) EmbeddingClustering.Normalize(center);
        }
        var turns = new List<Turn>();
        foreach (var chunk in chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] active = Enumerable.Range(0, 3).Where(i => chunk.Windows[i] >= 0).ToArray();
            int[] assignment = Assign(active.Select(i => windows[chunk.Windows[i]].Embedding).ToArray(), centers);
            turns.AddRange(FrameTurns(chunk.Offset, chunk.Length, chunk.Activity, active, assignment));
        }
        var merged = new List<Turn>();
        foreach (var group in turns.GroupBy(t => t.Speaker.Id))
        {
            Turn? previous = null;
            foreach (var turn in group.OrderBy(t => t.Start))
            {
                if (previous is not null && turn.Start <= previous.End) previous = previous with { End = previous.End > turn.End ? previous.End : turn.End };
                else { if (previous is not null) merged.Add(previous); previous = turn; }
            }
            if (previous is not null) merged.Add(previous);
        }
        return [.. merged.OrderBy(t => t.Start).ThenBy(t => t.Speaker.Id, StringComparer.Ordinal)];
    }

    internal static ImmutableArray<Turn> FrameTurns(long offset, int length, int[] activity, int[] active, int[] assignment)
    {
        var turns = ImmutableArray.CreateBuilder<Turn>();
        for (int s = 0; s < active.Length; s++)
        {
            int start = -1, mask = 1 << active[s];
            for (int frame = 0; frame <= PyannoteSegmenter.Frames; frame++)
            {
                int boundary = PyannoteSegmenter.Boundary(frame, length);
                bool on = frame < PyannoteSegmenter.Frames && boundary < length && (activity[frame] & mask) != 0;
                if (on && start < 0) start = boundary;
                if (!on && start >= 0)
                {
                    if (boundary > start) turns.Add(new Turn { Speaker = new SpeakerLabel { Id = "speaker-" + assignment[s] }, Start = TimeSpan.FromSeconds((offset + start) / 16000.0), End = TimeSpan.FromSeconds((offset + boundary) / 16000.0), Text = "" });
                    start = -1;
                }
            }
        }
        return turns.ToImmutable();
    }

    private static float[] Select(float[] samples, int length, int[] activity, int speaker, bool exclusive)
    {
        int mask = 1 << speaker, count = 0;
        for (int f = 0; f < activity.Length; f++)
            if ((activity[f] & mask) != 0 && (!exclusive || activity[f] == mask)) count += PyannoteSegmenter.Boundary(f + 1, length) - PyannoteSegmenter.Boundary(f, length);
        float[] result = new float[count]; int offset = 0;
        for (int f = 0; f < activity.Length; f++)
            if ((activity[f] & mask) != 0 && (!exclusive || activity[f] == mask))
            {
                int left = PyannoteSegmenter.Boundary(f, length), size = PyannoteSegmenter.Boundary(f + 1, length) - left;
                samples.AsSpan(left, size).CopyTo(result.AsSpan(offset)); offset += size;
            }
        return result;
    }

    // At most 3 local speakers and 32 global centers: exhaustive injective assignment is bounded by 32³.
    internal static int[] Assign(ReadOnlyMemory<float>[] points, float[][] centers)
    {
        if (points.Length > 3 || centers.Length < points.Length || centers.Length > 32) throw new ArgumentException("Invalid local/global speaker count.");
        int[] current = new int[points.Length], best = new int[points.Length]; double bestScore = double.NegativeInfinity;
        void Visit(int depth, uint used, double score)
        {
            if (depth == points.Length) { if (score > bestScore) { bestScore = score; current.CopyTo(best, 0); } return; }
            for (int c = 0; c < centers.Length; c++)
            {
                if ((used & (1u << c)) != 0) continue;
                double dot = 0;
                for (int d = 0; d < centers[c].Length; d++) dot += points[depth].Span[d] * (double)centers[c][d];
                current[depth] = c; Visit(depth + 1, used | (1u << c), score + dot);
            }
        }
        Visit(0, 0, 0); return best;
    }
}
