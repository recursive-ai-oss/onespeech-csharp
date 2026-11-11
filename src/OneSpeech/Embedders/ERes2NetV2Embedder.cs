using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Embedders;

/// <summary>ERes2NetV2 speaker embeddings (192 dimensions) from a locally converted, verified OneModel graph.
/// Instances serialize inference; dispose after consumers stop. No network access or persistence.</summary>
public sealed partial class ERes2NetV2Embedder : SpeakerEmbedder
{
    private readonly Session graph;
    private readonly SpeakerFbank features = new();
    private readonly object gate = new();
    private bool disposed;

    /// <summary>Loads a converted ERes2NetV2 graph. Model provenance is required; the dimension is fixed at 192.
    /// Defaults to one CPU thread. Throws ArgumentOutOfRangeException for nonpositive
    /// thread counts; file and model errors propagate from OneModel. Cancellation is checked on load.</summary>
    public ERes2NetV2Embedder(string graphPath, ModelInfo model, int threads = 1, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(graphPath);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
        Model = model;
        GraphTiming.OverrideThreads(ref threads);
        GraphTiming.BeginCost(0);
        graph = Session.Load(graphPath, new ModelOptions { IntraOpThreads = threads }, cancellationToken: cancellationToken);
        GraphTiming.EndCost(0);
    }
    /// <inheritdoc />
    public override ModelInfo Model { get; }
    /// <inheritdoc />
    public override int Dimension => 192;
    /// <inheritdoc />
    public override int SampleRate => 16000;

    /// <summary>Extracts an unnormalized speaker vector from mono 16,000 Hz PCM, 200 ms through 60 s.
    /// Destination must have exactly Dimension elements. Checks cancellation each feature frame and before
    /// graph inference. Rejects nonfinite audio or a nonfinite/zero model result. Never persists inputs/results.</summary>
    public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (audio.SampleRate != SampleRate || audio.Channels != 1) throw new ArgumentException("ERes2NetV2 requires mono 16000 Hz PCM.", nameof(audio));
            if (destination.Length != Dimension) throw new ArgumentException("Embedding destination has the wrong dimension.", nameof(destination));
            if (audio.Samples.Length < 3200) throw new ArgumentException("ERes2NetV2 pooling needs at least two frames after 8x down-sampling, so at least 200 ms of audio.", nameof(audio));
            cancellationToken.ThrowIfCancellationRequested();
            GraphTiming.BeginCost(1);
            float[] values = features.Compute(audio.Samples.Span, cancellationToken);
            GraphTiming.EndCost(1);
            using var input = Tensor.Zeros(DType.F32, Shape.Create(1, 80, values.Length / 80));
            values.CopyTo(input.MutableData<float>());
            cancellationToken.ThrowIfCancellationRequested();
            GraphTiming.Begin(1);
            using var output = graph.Run(new Dictionary<string, Tensor> { ["features"] = input });
            GraphTiming.End();
            cancellationToken.ThrowIfCancellationRequested();
            var vector = output["embedding"].Data<float>();
            if (vector.Length != Dimension) throw new InvalidDataException("ERes2NetV2 graph returned the wrong embedding dimension.");
            double norm = 0;
            foreach (float x in vector) { if (!float.IsFinite(x)) throw new InvalidDataException("ERes2NetV2 produced a nonfinite embedding."); norm += (double)x * x; }
            if (norm == 0) throw new InvalidDataException("ERes2NetV2 produced a zero embedding.");
            vector.CopyTo(destination);
        }
    }
    /// <inheritdoc />
    public override void Dispose() { lock (gate) { if (disposed) return; disposed = true; graph.Dispose(); } }
}
