using OneSpeech.Catalog.Conversion;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.FastConformer;

internal delegate void NemotronEncodedChunk(ReadOnlySpan<float> encoded, int validFrames, CancellationToken cancellationToken);

// Caller-owned fixed-size tensors keep both cache graphs on two warmed shape plans.
// State is private to an utterance; model sessions are borrowed and must outlive it.
internal sealed class NemotronAsrEncoderState : IDisposable
{
    private readonly FastConformerDimensions dimensions;
    private readonly int chunkFrames;
    private readonly NemotronEncodedChunk emit;
    private readonly List<Tensor> owned = [];
    private readonly List<Tensor> recurrent = [];
    private readonly Tensor initialFeatures, features, mask, encoded;
    private readonly Binding firstInitial, firstRegular, second;
    private bool firstChunk = true;
    private bool completed;
    private int pending, pastFrames;
    internal int EncodedFrames { get; private set; }

    internal NemotronAsrEncoderState(Session first, Session last, FastConformerDimensions dimensions, int chunkFrames, NemotronEncodedChunk emit, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(first); ArgumentNullException.ThrowIfNull(last); ArgumentNullException.ThrowIfNull(emit);
        dimensions.Validate();
        if (dimensions.Layers % 2 != 0 || chunkFrames is not (1 or 2 or 4 or 7 or 14)) throw new ArgumentException("Expected an even layer count and a supported chunk size.");
        this.dimensions = dimensions; this.chunkFrames = chunkFrames; this.emit = emit;
        try
        {
            int width = dimensions.Width, heads = dimensions.Heads, layers = dimensions.Layers, half = layers / 2;
            initialFeatures = Float(1, dimensions.MelBins, chunkFrames * 8 - 7);
            features = Float(1, dimensions.MelBins, chunkFrames * 8);
            mask = Float(1, 1, chunkFrames, 56 + chunkFrames);
            var middle = Float(1, width, chunkFrames); encoded = Float(1, width, chunkFrames);
            var initialSubsampling = new Tensor[3]; var subsampling = new Tensor[3];
            int frequency = dimensions.MelBins;
            for (int stage = 0; stage < 3; stage++)
            {
                int channels = stage == 0 ? 1 : dimensions.SubsamplingChannels;
                initialSubsampling[stage] = Float(1, channels, 2, frequency);
                subsampling[stage] = Float(1, channels, 1, frequency); recurrent.Add(subsampling[stage]); frequency = frequency / 2 + 1;
            }
            var keys = new Tensor[layers]; var values = new Tensor[layers]; var convolutions = new Tensor[layers];
            for (int layer = 0; layer < layers; layer++)
            {
                keys[layer] = Float(1, heads, 56, width / heads); values[layer] = Float(1, heads, 56, width / heads);
                convolutions[layer] = Float(1, width, dimensions.ConvolutionKernel - 1);
                recurrent.Add(keys[layer]); recurrent.Add(values[layer]); recurrent.Add(convolutions[layer]);
            }
            Binding Bind(Session session, Tensor input, Tensor output, int begin, int count, Tensor[]? subsamplingInputs)
            {
                List<string> inputNames = ["features", "attention_mask"], outputNames = ["encoded"];
                List<Tensor> inputs = [input, mask], outputs = [output];
                if (subsamplingInputs != null)
                    for (int stage = 0; stage < 3; stage++)
                    {
                        inputNames.Add("subsampling_cache" + stage); inputs.Add(subsamplingInputs[stage]);
                        outputNames.Add("next_subsampling_cache" + stage); outputs.Add(subsampling[stage]);
                    }
                for (int layer = begin; layer < begin + count; layer++)
                    foreach (var (kind, tensor) in new (string, Tensor)[] { ("key_cache", keys[layer]), ("value_cache", values[layer]), ("convolution_cache", convolutions[layer]) })
                    {
                        inputNames.Add(kind + layer); inputs.Add(tensor); outputNames.Add("next_" + kind + layer); outputs.Add(tensor);
                    }
                return new(session, inputNames.ToArray(), inputs.ToArray(), outputNames.ToArray(), outputs.ToArray());
            }
            firstInitial = Bind(first, initialFeatures, middle, 0, half, initialSubsampling);
            firstRegular = Bind(first, features, middle, 0, half, subsampling);
            second = Bind(last, middle, encoded, half, half, null);
            SetMask(chunkFrames); firstInitial.Run(cancellationToken); second.Run(cancellationToken);
            pastFrames = 56; SetMask(chunkFrames); firstRegular.Run(cancellationToken); second.Run(cancellationToken);
            Reset();
        }
        catch { foreach (var tensor in owned) tensor.Dispose(); owned.Clear(); throw; }
    }

    internal void Append(ReadOnlySpan<float> frame, CancellationToken cancellationToken)
    {
        if (completed) throw new InvalidOperationException("Encoder state is complete; reset before reuse.");
        if (frame.Length != dimensions.MelBins) throw new ArgumentException("Wrong mel frame width.", nameof(frame));
        cancellationToken.ThrowIfCancellationRequested();
        int capacity = firstChunk ? chunkFrames * 8 - 7 : chunkFrames * 8;
        var input = firstChunk ? initialFeatures.MutableData<float>() : features.MutableData<float>();
        if (pending == 0) input.Clear();
        for (int channel = 0; channel < frame.Length; channel++) input[channel * capacity + pending] = frame[channel];
        pending++;
        if (pending == capacity) RunChunk(chunkFrames, cancellationToken);
    }

    internal void Complete(CancellationToken cancellationToken)
    {
        if (completed) throw new InvalidOperationException("Encoder state is complete; reset before reuse.");
        cancellationToken.ThrowIfCancellationRequested();
        int valid = firstChunk ? (pending + 7) / 8 : pending / 8;
        if (valid > 0) RunChunk(valid, cancellationToken);
        completed = true;
    }

    private void RunChunk(int validFrames, CancellationToken cancellationToken)
    {
        SetMask(validFrames);
        (firstChunk ? firstInitial : firstRegular).Run(cancellationToken); second.Run(cancellationToken);
        emit(encoded.Data<float>(), validFrames, cancellationToken);
        EncodedFrames += validFrames; pastFrames = Math.Min(56, pastFrames + validFrames);
        firstChunk = false; pending = 0;
    }

    private void SetMask(int validFrames)
    {
        var values = mask.MutableData<float>();
        int keys = 56 + chunkFrames;
        for (int query = 0; query < chunkFrames; query++)
            for (int key = 0; key < keys; key++) values[query * keys + key] = key < 56 - pastFrames || key >= 56 + validFrames ? float.NegativeInfinity : 0;
    }

    internal void Reset()
    {
        foreach (var tensor in recurrent) tensor.MutableData<float>().Clear();
        initialFeatures.MutableData<float>().Clear(); features.MutableData<float>().Clear();
        firstChunk = true; completed = false; pending = pastFrames = EncodedFrames = 0;
    }

    private Tensor Float(params int[] shape)
    {
        var tensor = Tensor.Zeros(DType.F32, Shape.Create(shape)); owned.Add(tensor); return tensor;
    }
    public void Dispose() { foreach (var tensor in owned) tensor.Dispose(); owned.Clear(); recurrent.Clear(); }

    private sealed record Binding(Session Session, string[] InputNames, Tensor[] Inputs, string[] OutputNames, Tensor[] Outputs)
    {
        internal void Run(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested(); Session.Run(InputNames, Inputs, OutputNames, Outputs); cancellationToken.ThrowIfCancellationRequested();
        }
    }
}
