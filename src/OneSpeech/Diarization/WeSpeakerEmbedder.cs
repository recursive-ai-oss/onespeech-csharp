using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using OneSpeech.Embedders;

namespace OneSpeech.Diarization;

/// <summary>WeSpeaker ResNet34 speaker encoder (Wang et al. 2023) as published for the community pipeline: 80-band
/// Kaldi-style filterbanks of 16-bit-scaled PCM, a ResNet34 frame encoder, activity-weighted statistics pooling and a
/// 256-element linear layer. Owns its graph; calls are serialized. Internal: reached through CommunityDiarizer.</summary>
internal sealed class WeSpeakerEmbedder : IDisposable
{
    internal const int Dimension = 256, FrameChannels = 2560;
    private readonly Session graph;
    private readonly SpeakerFbank features = new(hamming: true);
    private readonly float[] projection, bias;
    private readonly object gate = new();
    private bool disposed;

    internal WeSpeakerEmbedder(string graphPath, float[] projection, float[] bias, int threads, CancellationToken cancellationToken)
    {
        if (projection.Length != Dimension * 2 * FrameChannels || bias.Length != Dimension) throw new ArgumentException("Unexpected projection size.");
        this.projection = projection; this.bias = bias;
        graph = Session.Load(graphPath, new ModelOptions { IntraOpThreads = threads }, cancellationToken: cancellationToken);
    }

    /// <summary>Runs the ResNet on the filterbank of <paramref name="samples"/> (mono 16 kHz, floats in [-1, 1]).
    /// Returns [2560, steps] row-major frame features; steps = ceil(frames / 8).</summary>
    internal float[] Frames(ReadOnlySpan<float> samples, out int steps, CancellationToken cancellationToken)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            // The published model was trained on int16-scaled PCM (observed: embeddings match only with this scale).
            float[] scaled = new float[samples.Length];
            for (int i = 0; i < scaled.Length; i++) scaled[i] = samples[i] * 32768f;
            float[] values = features.Compute(scaled, cancellationToken);
            using var input = Tensor.Zeros(DType.F32, Shape.Create(1, 80, values.Length / 80));
            values.CopyTo(input.MutableData<float>());
            using var output = graph.Run(new Dictionary<string, Tensor> { ["features"] = input });
            var frames = output["frames"].Data<float>();
            steps = frames.Length / FrameChannels;
            return frames.ToArray();
        }
    }

    /// <summary>Embedding of one clip, optionally weighting frames; weights are resampled to the encoder steps.</summary>
    internal float[] Embed(ReadOnlySpan<float> samples, ReadOnlySpan<float> weights, CancellationToken cancellationToken)
    {
        float[] frames = Frames(samples, out int steps, cancellationToken);
        return Project(Pool(frames, steps, weights));
    }

    // Weighted unbiased variance with reliability weights: sum(w (x-mean)^2) / (sum(w) - sum(w^2)/sum(w)); equals the usual n-1 form for unit weights. Observed to match the reference output.
    internal static float[] Pool(float[] frames, int steps, ReadOnlySpan<float> weights)
    {
        var w = new double[steps];
        if (weights.IsEmpty) Array.Fill(w, 1.0);
        // Frame weights are mapped to encoder steps by nearest lower index (observed: this matched the reference's
        // embeddings to cosine 0.999999 where linear interpolation did not).
        else for (int t = 0; t < steps; t++) w[t] = weights[Math.Min(weights.Length - 1, (int)((long)t * weights.Length / steps))];
        double sum = 0, sumSquares = 0; foreach (double x in w) { sum += x; sumSquares += x * x; }
        var pooled = new float[2 * FrameChannels];
        for (int c = 0; c < FrameChannels; c++)
        {
            double mean = 0;
            for (int t = 0; t < steps; t++) mean += w[t] * frames[c * steps + t];
            mean /= sum;
            double variance = 0;
            for (int t = 0; t < steps; t++) { double d = frames[c * steps + t] - mean; variance += w[t] * d * d; }
            variance /= sum - sumSquares / sum;
            pooled[c] = (float)mean;
            pooled[FrameChannels + c] = (float)Math.Sqrt(variance + 1e-7);
        }
        return pooled;
    }

    internal float[] Project(float[] pooled)
    {
        var result = new float[Dimension];
        for (int o = 0; o < Dimension; o++)
        {
            double s = bias[o];
            int row = o * pooled.Length;
            for (int i = 0; i < pooled.Length; i++) s += projection[row + i] * (double)pooled[i];
            result[o] = (float)s;
        }
        return result;
    }

    public void Dispose() { lock (gate) { if (disposed) return; disposed = true; graph.Dispose(); } }
}
