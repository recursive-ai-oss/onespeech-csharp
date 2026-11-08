using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Whisper;

internal sealed class WhisperModelSession : IDisposable
{
    private readonly Session encoder;
    private readonly Session? encoderPart;
    private readonly Session decoder;
    internal WhisperConfiguration Configuration { get; }
    internal WhisperModelSession(string directory, int threads, CancellationToken cancellationToken)
    {
        Configuration = WhisperConfiguration.Load(directory);
        var options = new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 };
        if (File.Exists(Path.Combine(directory, "encoder-part0.onemodel"))) encoderPart = Session.Load(Path.Combine(directory, "encoder-part0.onemodel"), options, cancellationToken: cancellationToken);
        try { encoder = Session.Load(Path.Combine(directory, "encoder.onemodel"), options, cancellationToken: cancellationToken); }
        catch { encoderPart?.Dispose(); throw; }
        try { decoder = Session.Load(Path.Combine(directory, "decoder.onemodel"), options, cancellationToken: cancellationToken); }
        catch { encoder.Dispose(); encoderPart?.Dispose(); throw; }
    }

    internal SessionOutputs Encode(float[] features, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var input = Tensor.FromSpan<float>(features, Shape.Create(1, Configuration.MelBins, 3000));
        using var intermediate = encoderPart?.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["mel"] = input });
        var result = intermediate == null
            ? encoder.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["mel"] = input })
            : encoder.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["hidden"] = intermediate["encoder"] });
        if (cancellationToken.IsCancellationRequested) { result.Dispose(); cancellationToken.ThrowIfCancellationRequested(); }
        return result;
    }

    internal SessionOutputs Decode(SessionOutputs encoded, ReadOnlySpan<int> tokens, SessionOutputs? previous, int pastLength, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int length = tokens.Length;
        if (length <= 0 || pastLength < 0 || pastLength + length > 448) throw new ArgumentOutOfRangeException(nameof(tokens));
        long[] tokenValues = new long[length], positions = new long[length];
        for (int i = 0; i < length; i++) { tokenValues[i] = tokens[i]; positions[i] = pastLength + i; }
        float[] mask = new float[length * (pastLength + length)];
        for (int q = 0; q < length; q++)
            for (int k = pastLength + q + 1; k < pastLength + length; k++) mask[q * (pastLength + length) + k] = float.NegativeInfinity;
        using var tokenInput = Tensor.FromSpan<long>(tokenValues, Shape.Create(1, length));
        using var positionInput = Tensor.FromSpan<long>(positions, Shape.Create(length));
        using var maskInput = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, length, pastLength + length));
        using var empty = Tensor.Create(DType.F32, Shape.Create(1, Configuration.DecoderHeads, 0, Configuration.Width / Configuration.DecoderHeads));
        var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["tokens"] = tokenInput, ["positions"] = positionInput, ["mask"] = maskInput };
        for (int layer = 0; layer < Configuration.DecoderLayers; layer++)
        {
            inputs["past_k_" + layer] = previous == null ? empty : previous["present_k_" + layer];
            inputs["past_v_" + layer] = previous == null ? empty : previous["present_v_" + layer];
            inputs["cross_k_" + layer] = encoded["cross_k_" + layer];
            inputs["cross_v_" + layer] = encoded["cross_v_" + layer];
        }
        var result = decoder.Run(inputs);
        if (cancellationToken.IsCancellationRequested) { result.Dispose(); cancellationToken.ThrowIfCancellationRequested(); }
        return result;
    }
    public void Dispose() { decoder.Dispose(); encoder.Dispose(); encoderPart?.Dispose(); }
}
