using System.Collections.Immutable;

namespace OneSpeech.Custom;

/// <summary>A supplied graph or vocabulary does not satisfy the declared speech-model contract.</summary>
public sealed class InvalidModelShapeException : OneSpeechException
{
    /// <summary>Creates an actionable shape error without executing or modifying weights.</summary>
    public InvalidModelShapeException(string message) : base(message) { }
}

/// <summary>Mono 16000 Hz front ends. No resampling or waveform normalization is implicit.</summary>
public enum GraphFrontEnd
{
    /// <summary>Unmodified waveform, shape [1,1,samples].</summary>
    RawWaveform,
    /// <summary>Whisper 80-bin log-mel, unpadded, shape [1,80,samples/160].</summary>
    Whisper80,
    /// <summary>Whisper 128-bin log-mel, unpadded, shape [1,128,samples/160].</summary>
    Whisper128,
    /// <summary>NeMo 128-bin normalized log-mel, with its masked final STFT frame removed.</summary>
    Nemo128
}

/// <summary>One float32 input/output encoder stage. Inputs are [1,channels,frames]; output has the declared channel count. Only the final CTC stage may be frame-major [1,frames,vocabulary]. Batch size is always one.</summary>
public sealed record EncoderGraph
{
    /// <summary>Local ONNX or OneModel file. External ONNX data must be alongside the file. No download occurs.</summary>
    public required string Path { get; init; }
    /// <summary>Output channels (vocabulary size for CTC), in 1..1000000.</summary>
    public required int Channels { get; init; }
    /// <summary>Temporal stride, in input frames; default 1, range 1..16000.</summary>
    public int Stride { get; init; } = 1;
    /// <summary>Receptive field in input frames for floor((frames-ReceptiveField)/Stride)+1; default 1. Set 400 with stride 320 for raw wav2vec2.</summary>
    public int ReceptiveField { get; init; } = 1;
    /// <summary>If true use ceiling(frames/Stride) instead, for same-padded subsampling. Default false.</summary>
    public bool SamePadding { get; init; }
}

/// <summary>Explicit packed recurrent predictor/joint contract for RNN-T or TDT. All tensors are float32, batch one; caller-owned inputs are copied at load.</summary>
public sealed record TransducerGraphs
{
    /// <summary>Predictor graph: [1,StateSize] to [1,StateSize]. The first EmbeddingSize entries are replaced by the accepted token embedding before each recurrent step. The initial state is zero and is run once for blank priming.</summary>
    public required string PredictorPath { get; init; }
    /// <summary>Joint graph: [1,encoderChannels+EmbeddingSize] to [1,vocabularySize+Durations.Length]. Encoder frame precedes the predictor output's first EmbeddingSize entries.</summary>
    public required string JointPath { get; init; }
    /// <summary>Packed predictor state size, default 3200; range EmbeddingSize..1000000.</summary>
    public int StateSize { get; init; } = 3200;
    /// <summary>Token embedding and predictor output width, default 640; range 1..65536.</summary>
    public int EmbeddingSize { get; init; } = 640;
    /// <summary>Row-major vocabularySize by EmbeddingSize token embedding table, including blank.</summary>
    public required ReadOnlyMemory<float> Embeddings { get; init; }
    /// <summary>Empty for RNN-T; TDT duration-head frame advances, including zero and a positive duration. Default empty.</summary>
    public ImmutableArray<int> Durations { get; init; } = [];
    /// <summary>Maximum emissions without advancing a frame; default 10, range 1..1024.</summary>
    public int MaxSymbolsPerFrame { get; init; } = 10;
}
