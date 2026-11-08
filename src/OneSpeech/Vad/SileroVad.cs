using System.Collections.Immutable;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Vad;

/// <summary>Silero v6.2.3 speech detection. Accepts mono 8 or 16 kHz PCM; instances own the graph and must be disposed.</summary>
public sealed class SileroVad : VoiceActivityDetector, IDisposable
{
    private readonly Session graph;
    private readonly SileroVadOptions options;

    /// <summary>Loads the embedded, SHA-256-verified ONNX model. Defaults to one inference thread and standard segmentation options. Cancellation is checked while loading.</summary>
    public SileroVad(SileroVadOptions? options = null, CancellationToken cancellationToken = default)
    {
        this.options = options ?? new();
        this.options.Validate();
        using var stream = typeof(SileroVad).Assembly.GetManifestResourceStream("OneSpeech.silero_vad.onnx")
            ?? throw new ModelNotFoundException("The embedded Silero VAD model is missing. Rebuild with the pinned SileroModelPath.");
        GraphTiming.BeginCost(0);
        graph = Session.Load(stream, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: cancellationToken);
        GraphTiming.EndCost(0);
    }

    /// <inheritdoc />
    public override ModelInfo Model { get; } = new()
    {
        Id = "silero/vad-v6.2.3", Source = "https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx",
        Revision = "5cd7945676eb32225748052e2e6a0580e4686a08", License = "MIT", Tier = ModelTier.Tier1,
        Attribution = "Copyright (c) 2020-present Silero Team",
        Files = [new() { Name = "silero_vad.onnx", Sha256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3" }]
    };

    /// <summary>Creates independent recurrent state at 8 or 16 kHz. The detector must outlive the state. Throws ArgumentOutOfRangeException for other rates.</summary>
    public SileroVadState CreateState(int sampleRate) => new(graph, sampleRate);

    /// <summary>Detects intervals using 32 ms frames, zero-padding the last frame. Resets recurrence for each call; cancellation is checked every frame.</summary>
    public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        cancellationToken.ThrowIfCancellationRequested();
        if (audio.SampleRate is not (8000 or 16000) || audio.Channels != 1) throw new ArgumentException("Silero VAD audio requires mono PCM at 8000 or 16000 Hz.", nameof(audio));
        using var state = CreateState(audio.SampleRate);
        var segments = new SpeechSegmenter(audio.SampleRate, options);
        var raw = new List<SpeechInterval>();
        var frame = new float[state.FrameSamples];
        for (int start = 0; start < audio.FrameCount; start += frame.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            frame.AsSpan().Clear();
            audio.Samples.Span.Slice(start, Math.Min(frame.Length, audio.FrameCount - start)).CopyTo(frame);
            if (segments.Push(state.ProcessFrame(frame, cancellationToken)) is { } interval) raw.Add(interval);
        }
        if (segments.Finish(audio.FrameCount) is { } last) raw.Add(last);
        return SpeechSegmenter.Pad(raw, audio.Duration, options.SpeechPadding);
    }

    /// <summary>Releases graph resources. Do not dispose during a call or while a state is in use.</summary>
    protected override void Dispose(bool disposing) { try { if (disposing) graph.Dispose(); } finally { base.Dispose(disposing); } }
}

/// <summary>Independent recurrent Silero state. Not thread-safe; one frame must be exactly 512 samples at 16 kHz or 256 at 8 kHz.</summary>
public sealed class SileroVadState : IDisposable
{
    private readonly Session graph;
    private readonly Tensor input;
    private readonly Tensor state;
    private readonly Tensor rate;
    private static readonly string[] InputNames = ["input", "state", "sr"];
    private static readonly string[] OutputNames = ["stateN", "output"];
    private readonly Tensor[] inputs;
    private readonly Tensor[] outputs;
    private readonly int context;

    internal SileroVadState(Session graph, int sampleRate)
    {
        if (sampleRate is not (8000 or 16000)) throw new ArgumentOutOfRangeException(nameof(sampleRate), "Silero supports 8000 and 16000 Hz.");
        this.graph = graph;
        FrameSamples = sampleRate / 1000 * 32;
        context = sampleRate / 250;
        input = Tensor.Zeros(DType.F32, Shape.Create(1, FrameSamples + context));
        state = Tensor.Zeros(DType.F32, Shape.Create(2, 1, 128));
        rate = Tensor.Scalar((long)sampleRate);
        inputs = [input, state, rate];
        outputs = [Tensor.Zeros(DType.F32, Shape.Create(2, 1, 128)), Tensor.Zeros(DType.F32, Shape.Create(1, 1))];
    }

    /// <summary>Required mono frame size in samples: 512 at 16 kHz, 256 at 8 kHz.</summary>
    public int FrameSamples { get; }

    /// <summary>Returns speech probability in [0,1], updating recurrence and context. Throws ArgumentException for an incorrect frame length; checks cancellation before inference.</summary>
    public float ProcessFrame(ReadOnlySpan<float> frame, CancellationToken cancellationToken)
    {
        if (frame.Length != FrameSamples) throw new ArgumentException("Expected one complete 32 ms frame.", nameof(frame));
        cancellationToken.ThrowIfCancellationRequested();
        frame.CopyTo(input.MutableData<float>()[context..]);
        GraphTiming.Begin(2);
        graph.Run(InputNames, inputs, OutputNames, outputs);
        GraphTiming.End();
        outputs[0].Data<float>().CopyTo(state.MutableData<float>());
        frame[^context..].CopyTo(input.MutableData<float>());
        return outputs[1].Data<float>()[0];
    }

    /// <summary>Resets recurrence and the previous-frame context before a new recording.</summary>
    public void Reset()
    {
        input.MutableData<float>().Clear();
        state.MutableData<float>().Clear();
    }

    /// <summary>Releases owned input and output tensors. Does not dispose the shared graph.</summary>
    public void Dispose() { foreach (var output in outputs) output.Dispose(); input.Dispose(); state.Dispose(); rate.Dispose(); }
}
