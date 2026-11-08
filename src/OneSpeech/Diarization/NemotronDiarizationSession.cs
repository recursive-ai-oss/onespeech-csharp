using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Diarization;

internal sealed class NemotronDiarizationSession : IDisposable
{
    private readonly Session embedding;
    private readonly Session model;
    internal NemotronDiarizationSession(string directory, int threads, CancellationToken cancellationToken)
    {
        var options = new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 };
        embedding = Session.Load(Path.Combine(directory, "embedding.onemodel"), options, cancellationToken: cancellationToken);
        try { model = Session.Load(Path.Combine(directory, "diarization.onemodel"), options, cancellationToken: cancellationToken); }
        catch { embedding.Dispose(); throw; }
    }
    internal float[] Embed(float[] stacked, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var tensor = Tensor.FromSpan<float>(stacked, Shape.Create(1, stacked.Length / 1024, 1024));
        using var output = embedding.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["stacked"] = tensor });
        return output["embedded"].Data<float>().ToArray();
    }
    internal (float[] Probabilities, float[] Encoded, float[] Upsampled) Forward(float[] embedded, CancellationToken cancellationToken, bool diagnostics = false, int? validFrames = null)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int frames = embedded.Length / 512;
        if (frames is < 1 or > 1024 || embedded.Length != frames * 512) throw new ArgumentException("Expected 1..1024 embedded frames.");
        int valid = validFrames ?? frames;
        if (valid < 1 || valid > frames) throw new ArgumentOutOfRangeException(nameof(validFrames));
        float[] mask = new float[frames];
        Array.Fill(mask, float.NegativeInfinity, valid, frames - valid);
        float[] cos = new float[frames * 64], sin = new float[cos.Length];
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < 32; i++)
            {
                float phase = f * MathF.Pow(10000, -2f * i / 64);
                cos[f * 64 + i] = cos[f * 64 + i + 32] = MathF.Cos(phase);
                sin[f * 64 + i] = sin[f * 64 + i + 32] = MathF.Sin(phase);
            }
        using var input = Tensor.FromSpan<float>(embedded, Shape.Create(1, frames, 512));
        using var cosTensor = Tensor.FromSpan<float>(cos, Shape.Create(1, 1, frames, 64));
        using var sinTensor = Tensor.FromSpan<float>(sin, Shape.Create(1, 1, frames, 64));
        using var maskTensor = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, 1, frames));
        using var output = model.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["embedded"] = input, ["cos"] = cosTensor, ["sin"] = sinTensor, ["mask"] = maskTensor });
        cancellationToken.ThrowIfCancellationRequested();
        return (output["probabilities"].Data<float>().ToArray(), diagnostics ? output["encoded"].Data<float>().ToArray() : [], diagnostics ? output["upsampled"].Data<float>().ToArray() : []);
    }
    public void Dispose() { model.Dispose(); embedding.Dispose(); }
}
