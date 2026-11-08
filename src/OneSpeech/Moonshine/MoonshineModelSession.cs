using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Moonshine;

internal sealed class MoonshineModelSession : IDisposable
{
    internal MoonshineConfiguration Configuration { get; }
    private readonly Session encoder;
    private readonly Session decoder;
    internal MoonshineModelSession(string directory, int threads, CancellationToken cancellationToken)
    {
        Configuration = MoonshineConfiguration.Load(directory);
        encoder = Session.Load(Path.Combine(directory, "encoder.onemodel"), new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 }, cancellationToken: cancellationToken);
        try { decoder = Session.Load(Path.Combine(directory, "decoder.onemodel"), new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 }, cancellationToken: cancellationToken); }
        catch { encoder.Dispose(); throw; }
    }
    internal SessionOutputs Encode(AudioBuffer audio, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var c = Configuration;
        // Released public outputs retain the padded frame count but mask an incomplete
        // 80-sample frame in its entirety before the frontend convolutions.
        int count = c.Streaming ? (audio.FrameCount + 79) / 80 * 80 : audio.FrameCount;
        int frames = c.Streaming ? (count / 80 + 3) / 4 : (((count - 127) / 64 + 1 - 7) / 3 + 1 - 3) / 2 + 1;
        using var input = Tensor.Zeros(DType.F32, Shape.Create(1, 1, count));
        audio.Samples.Span[..(c.Streaming ? audio.FrameCount / 80 * 80 : audio.FrameCount)].CopyTo(input.MutableData<float>());
        using var cos = Rotary(frames, c.RotaryDimension, false);
        using var sin = Rotary(frames, c.RotaryDimension, true);
        using var local = LocalMask(frames, 3);
        using var causal = LocalMask(frames, 0);
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["audio"] = input };
        if (c.Streaming) { inputs["local_mask"] = local; inputs["causal_mask"] = causal; }
        else { inputs["cos"] = cos; inputs["sin"] = sin; }
        return encoder.Run(inputs);
    }
    internal SessionOutputs Decode(Tensor encoded, ReadOnlySpan<int> ids, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int count = ids.Length;
        using var tokens = Tensor.FromSpan<long>(ids.ToArray().Select(x => (long)x).ToArray(), Shape.Create(1, count));
        using var cos = Rotary(count, Configuration.RotaryDimension, false);
        using var sin = Rotary(count, Configuration.RotaryDimension, true);
        float[] mask = new float[count * count];
        for (int q = 0; q < count; q++) for (int k = q + 1; k < count; k++) mask[q * count + k] = float.NegativeInfinity;
        using var attention = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, count, count));
        int frames = encoded.Data<float>().Length / Configuration.EncoderWidth;
        using var positions = Tensor.FromSpan<long>(Enumerable.Range(0, frames).Select(i => (long)i).ToArray(), Shape.Create(frames));
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["tokens"] = tokens, ["encoder"] = encoded, ["cos"] = cos, ["sin"] = sin, ["mask"] = attention };
        if (Configuration.Streaming) inputs["encoder_positions"] = positions;
        return decoder.Run(inputs);
    }
    private static Tensor LocalMask(int count, int right)
    {
        var data = new float[count * count];
        for (int q = 0; q < count; q++) for (int k = 0; k < count; k++) if (k < q - 15 || k > q + right) data[q * count + k] = float.NegativeInfinity;
        return Tensor.FromSpan<float>(data, Shape.Create(1, 1, count, count));
    }
    internal static Tensor Rotary(int count, int dimension, bool sine)
    {
        var data = new float[count * dimension];
        for (int p = 0; p < count; p++) for (int d = 0; d < dimension; d++)
        {
            float angle = p * (float)Math.Pow(10000, -2d * (d / 2) / dimension);
            data[p * dimension + d] = sine ? MathF.Sin(angle) : MathF.Cos(angle);
        }
        return Tensor.FromSpan<float>(data, Shape.Create(1, 1, count, dimension));
    }
    public void Dispose() { decoder.Dispose(); encoder.Dispose(); }
}
