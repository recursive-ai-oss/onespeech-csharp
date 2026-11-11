using OneSpeech.Formats;

namespace OneSpeech.Diarization;

/// <summary>The published x-vector transform and PLDA parameters of the community pipeline, applied as released
/// (nothing is fitted). Landini et al. 2022 §2.3: centre, whiten and length-normalise the embeddings, then rotate
/// them into the space where the within-speaker covariance is the identity and the between-speaker covariance is
/// diagonal (<see cref="Psi"/>). The order of operations is checked against the reference pipeline's clustering output.</summary>
internal sealed class PldaTransform
{
    private readonly double[] mean1, mean2, mu;
    private readonly double[,] lda, tr;
    internal double[] Psi { get; }
    internal const int InputDimension = 256, OutputDimension = 128;
    // Observed against the reference clustering: see docs/diarization.md.
    internal static bool TransposeRotation = true;

    private PldaTransform(double[] mean1, double[] mean2, double[,] lda, double[] mu, double[,] tr, double[] psi)
    { this.mean1 = mean1; this.mean2 = mean2; this.lda = lda; this.mu = mu; this.tr = tr; Psi = psi; }

    /// <summary>Reads plda.npz and xvec_transform.npz from the publisher snapshot. Throws InvalidDataException on any shape or content surprise.</summary>
    internal static PldaTransform Load(string pldaNpz, string transformNpz)
    {
        Dictionary<string, NpyArray> plda, xt;
        using (var f = File.OpenRead(pldaNpz)) plda = NpzArchive.Read(f);
        using (var f = File.OpenRead(transformNpz)) xt = NpzArchive.Read(f);
        return Create(plda, xt);
    }

    internal static PldaTransform Create(Dictionary<string, NpyArray> plda, Dictionary<string, NpyArray> xt)
    {
        NpyArray Get(Dictionary<string, NpyArray> d, string key, params int[] shape)
        {
            if (!d.TryGetValue(key, out var a) || !a.Shape.AsSpan().SequenceEqual(shape)) throw new InvalidDataException("Unexpected PLDA array " + key + ".");
            foreach (double v in a.Values) if (!double.IsFinite(v)) throw new InvalidDataException("Nonfinite PLDA value in " + key + ".");
            return a;
        }
        var lda = Get(xt, "lda", InputDimension, OutputDimension);
        var tr = Get(plda, "tr", OutputDimension, OutputDimension);
        var psi = Get(plda, "psi", OutputDimension);
        foreach (double p in psi.Values) if (p <= 0) throw new InvalidDataException("PLDA eigenvalues must be positive.");
        double[,] L = new double[InputDimension, OutputDimension], T = new double[OutputDimension, OutputDimension];
        for (int i = 0; i < InputDimension; i++) for (int j = 0; j < OutputDimension; j++) L[i, j] = lda.Values[i * OutputDimension + j];
        for (int i = 0; i < OutputDimension; i++) for (int j = 0; j < OutputDimension; j++) T[i, j] = tr.Values[i * OutputDimension + j];
        return new PldaTransform(Get(xt, "mean1", InputDimension).Values, Get(xt, "mean2", OutputDimension).Values, L, Get(plda, "mu", OutputDimension).Values, T, psi.Values);
    }

    /// <summary>The intermediate 128-element LDA-space vector, length-normalised to unit norm: the space used for the initial clustering.</summary>
    internal double[] Whitened(ReadOnlySpan<float> embedding)
    {
        var x = new double[InputDimension];
        double norm = 0;
        for (int i = 0; i < InputDimension; i++) { x[i] = embedding[i] - mean1[i]; norm += x[i] * x[i]; }
        double scale = Math.Sqrt(InputDimension) / Math.Max(Math.Sqrt(norm), 1e-12);
        var y = new double[OutputDimension];
        norm = 0;
        for (int j = 0; j < OutputDimension; j++)
        {
            double s = 0;
            for (int i = 0; i < InputDimension; i++) s += x[i] * scale * lda[i, j];
            y[j] = s - mean2[j];
            norm += y[j] * y[j];
        }
        for (int j = 0; j < OutputDimension; j++) y[j] /= Math.Max(Math.Sqrt(norm), 1e-12);
        return y;
    }

    /// <summary>Maps one 256-element embedding to the 128-element PLDA space.</summary>
    internal double[] Apply(ReadOnlySpan<float> embedding)
    {
        if (embedding.Length != InputDimension) throw new ArgumentException("Embedding length.", nameof(embedding));
        var x = new double[InputDimension];
        double norm = 0;
        for (int i = 0; i < InputDimension; i++) { x[i] = embedding[i] - mean1[i]; norm += x[i] * x[i]; }
        double scale = Math.Sqrt(InputDimension) / Math.Max(Math.Sqrt(norm), 1e-12);
        var y = new double[OutputDimension];
        norm = 0;
        for (int j = 0; j < OutputDimension; j++)
        {
            double s = 0;
            for (int i = 0; i < InputDimension; i++) s += x[i] * scale * lda[i, j];
            y[j] = s - mean2[j];
            norm += y[j] * y[j];
        }
        scale = Math.Sqrt(OutputDimension) / Math.Max(Math.Sqrt(norm), 1e-12);
        var z = new double[OutputDimension];
        for (int j = 0; j < OutputDimension; j++)
        {
            double s = 0;
            for (int i = 0; i < OutputDimension; i++) s += (y[i] * scale - mu[i]) * (TransposeRotation ? tr[j, i] : tr[i, j]);
            z[j] = s;
        }
        return z;
    }
}
