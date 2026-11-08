using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Moonshine;

internal sealed class MoonshineModelSession : IDisposable
{
    internal MoonshineConfiguration Configuration { get; }
    private readonly Session encoder;
    private readonly Session decoder;
    private readonly Session? project;
    private readonly Session? step;
    private static readonly string[] LiveEncoderInputs = ["audio", "local_mask", "causal_mask"];
    private static readonly string[] EncoderOutput = ["encoder"];
    private static readonly string[] ProjectInputs = ["encoder", "encoder_positions"];
    internal MoonshineModelSession(string directory, int threads, CancellationToken cancellationToken)
    {
        GraphTiming.BeginCost(0);
        Configuration = MoonshineConfiguration.Load(directory);
        GraphTiming.OverrideThreads(ref threads);
        encoder = Session.Load(Path.Combine(directory, "encoder.onemodel"), new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 }, cancellationToken: cancellationToken);
        try { decoder = Session.Load(Path.Combine(directory, "decoder.onemodel"), new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 }, cancellationToken: cancellationToken); }
        catch { encoder.Dispose(); throw; }
        try
        {
            if (Configuration.Streaming)
            {
                project = Session.Load(Path.Combine(directory, "decoder-project.onemodel"), new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 5 }, cancellationToken: cancellationToken);
                step = Session.Load(Path.Combine(directory, "decoder-step.onemodel"), new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 5 }, cancellationToken: cancellationToken);
            }
        }
        catch { Dispose(); throw; }
        GraphTiming.EndCost(0);
    }
    internal void RunLiveEncoder(Tensor[] inputs, Tensor[] outputs) { GraphTiming.Begin(5); encoder.Run(LiveEncoderInputs, inputs, EncoderOutput, outputs); GraphTiming.End(); }
    internal void RunProject(Tensor[] inputs, string[] names, Tensor[] outputs) { GraphTiming.Begin(5); project!.Run(ProjectInputs, inputs, names, outputs); GraphTiming.End(); }
    internal void RunStep(string[] names, Tensor[] inputs, string[] outputNames, Tensor[] outputs) { GraphTiming.Begin(5); step!.Run(names, inputs, outputNames, outputs); GraphTiming.End(); }
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
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["audio"] = input };
        if (c.Streaming)
        {
            // This position-free encoder never consumes rotary tensors.
            using var local = LocalMask(frames, 3);
            using var causal = LocalMask(frames, 0);
            inputs["local_mask"] = local; inputs["causal_mask"] = causal;
            GraphTiming.Begin(5);
            var streamed = encoder.Run(inputs);
            GraphTiming.End();
            return streamed;
        }
        using var cos = Rotary(frames, c.RotaryDimension, false);
        using var sin = Rotary(frames, c.RotaryDimension, true);
        inputs["cos"] = cos; inputs["sin"] = sin;
        GraphTiming.Begin(5);
        var result = encoder.Run(inputs);
        GraphTiming.End();
        return result;
    }
    internal SessionOutputs Decode(Tensor encoded, ReadOnlySpan<int> ids, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int count = ids.Length;
        using var tokens = Tensor.Zeros(DType.I64, Shape.Create(1, count));
        var tokenData = tokens.MutableData<long>();
        for (int i = 0; i < count; i++) tokenData[i] = ids[i];
        using var cos = Rotary(count, Configuration.RotaryDimension, false);
        using var sin = Rotary(count, Configuration.RotaryDimension, true);
        using var attention = Tensor.Zeros(DType.F32, Shape.Create(1, 1, count, count));
        var mask = attention.MutableData<float>();
        for (int q = 0; q < count; q++) for (int k = q + 1; k < count; k++) mask[q * count + k] = float.NegativeInfinity;
        int frames = encoded.Data<float>().Length / Configuration.EncoderWidth;
        using var positions = Tensor.Zeros(DType.I64, Shape.Create(frames));
        var positionData = positions.MutableData<long>();
        for (int i = 0; i < frames; i++) positionData[i] = i;
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["tokens"] = tokens, ["encoder"] = encoded, ["cos"] = cos, ["sin"] = sin, ["mask"] = attention };
        if (Configuration.Streaming) inputs["encoder_positions"] = positions;
        GraphTiming.Begin(5);
        var result = decoder.Run(inputs);
        GraphTiming.End();
        return result;
    }
    private static Tensor LocalMask(int count, int right)
    {
        var tensor = Tensor.Zeros(DType.F32, Shape.Create(1, 1, count, count));
        var data = tensor.MutableData<float>();
        for (int q = 0; q < count; q++) for (int k = 0; k < count; k++) if (k < q - 15 || k > q + right) data[q * count + k] = float.NegativeInfinity;
        return tensor;
    }
    internal static Tensor Rotary(int count, int dimension, bool sine)
    {
        var tensor = Tensor.Zeros(DType.F32, Shape.Create(1, 1, count, dimension));
        var data = tensor.MutableData<float>();
        for (int p = 0; p < count; p++) for (int d = 0; d < dimension; d++)
        {
            float angle = p * (float)Math.Pow(10000, -2d * (d / 2) / dimension);
            data[p * dimension + d] = sine ? MathF.Sin(angle) : MathF.Cos(angle);
        }
        return tensor;
    }
    public void Dispose() { step?.Dispose(); project?.Dispose(); decoder.Dispose(); encoder.Dispose(); }
}
