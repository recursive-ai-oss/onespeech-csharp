using System.Collections.Immutable;
using OneSpeech.Audio;
using OneSpeech.Decoding;
using OneSpeech.Pipeline;
using OneSpeech.Vad;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Custom;

/// <summary>Low-level local float32 CTC/RNN-T/TDT graph adapter. Owns loaded graphs; not thread-safe. Graphs must use the explicit EncoderGraph and TransducerGraphs contracts. No model discovery, conversion, licensing inference or downloading is performed.</summary>
public sealed class GraphSpeechRecognizer : SpeechRecognizer, IDisposable
{
    private readonly List<Model> encoders = [];
    private readonly EncoderGraph[] stages;
    private readonly GraphFrontEnd frontEnd;
    private readonly TokenVocabulary vocabulary;
    private readonly int blank, beam;
    private readonly TimeSpan frameDuration;
    private readonly TransducerGraphs? transducer;
    private readonly Model? predictor, joint;
    private bool disposed;
    /// <summary>Caller-supplied provenance, including the caller's licence, source and attribution. Weights are never persisted by this adapter.</summary>
    public override ModelInfo Model { get; }
    /// <summary>Native frame timings only. Language hints label the output but never condition decoding. No biasing, translation or language identification.</summary>
    public override Capabilities Capabilities { get; } = new() { NativeTimestamps = true };

    /// <summary>Loads local encoder stages and optional packed transducer graphs. CTC final output must be frame-major logits with VocabularySize columns; intermediate and transducer encoder outputs are channel-major. FrameDuration is the final encoder frame interval on the audio clock. BeamSize 0 selects CTC greedy; 1..64 selects prefix beam (vocabulary at most 4096), with Viterbi timing. Transducers are greedy only. Threads default 1 (1..256). Invalid contracts throw InvalidModelShapeException; I/O and OneModel load errors propagate. Cancels at each load.</summary>
    public GraphSpeechRecognizer(IEnumerable<EncoderGraph> encoder, GraphFrontEnd frontEnd, TokenVocabulary vocabulary, int blankId, TimeSpan frameDuration, ModelInfo model, TransducerGraphs? transducer = null, int beamSize = 0, int threads = 1, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(encoder); ArgumentNullException.ThrowIfNull(vocabulary); ArgumentNullException.ThrowIfNull(model);
        cancellationToken.ThrowIfCancellationRequested();
        stages = encoder.Take(17).ToArray();
        if (stages.Length is < 1 or > 16 || stages.Any(s => s is null || string.IsNullOrWhiteSpace(s.Path) || s.Channels is < 1 or > 1000000 || s.Stride is < 1 or > 16000 || s.ReceptiveField is < 1 or > 1000000) || !Enum.IsDefined(frontEnd) || blankId < 0 || blankId >= vocabulary.Pieces.Length || frameDuration <= TimeSpan.Zero || frameDuration > TimeSpan.FromSeconds(1) || threads is < 1 or > 256 || beamSize is < 0 or > 64)
            throw new InvalidModelShapeException("Invalid encoder, front end, vocabulary, blank, frame duration, beam or thread contract.");
        if (transducer is null && (stages[^1].Channels != vocabulary.Pieces.Length || beamSize > 0 && vocabulary.Pieces.Length > 4096))
            throw new InvalidModelShapeException("CTC output channels must equal vocabulary size; prefix beam supports at most 4096 IDs.");
        if (transducer is not null && (beamSize != 0 || transducer.EmbeddingSize is < 1 or > 65536 || transducer.StateSize < transducer.EmbeddingSize || transducer.StateSize > 1000000 || transducer.Embeddings.Length != (long)vocabulary.Pieces.Length * transducer.EmbeddingSize || transducer.MaxSymbolsPerFrame is < 1 or > 1024 || transducer.Durations.IsDefault || transducer.Durations.Length > 256 || transducer.Durations.Any(d => d < 0) || transducer.Durations.Length > 0 && (!transducer.Durations.Contains(0) || !transducer.Durations.Any(d => d > 0))))
            throw new InvalidModelShapeException("Invalid packed predictor, embeddings, duration table or transducer search contract.");
        if (transducer is not null)
        {
            foreach (float value in transducer.Embeddings.Span) if (!float.IsFinite(value)) throw new InvalidModelShapeException("Embeddings must be finite.");
            for (int i = 1; i < transducer.Durations.Length; i++)
                if (transducer.Durations[i] <= transducer.Durations[i - 1] || transducer.Durations[i] > 1024) throw new InvalidModelShapeException("Durations must be strictly increasing from zero and at most 1024 frames.");
        }
        this.frontEnd = frontEnd; this.vocabulary = vocabulary; blank = blankId; beam = beamSize; this.frameDuration = frameDuration; Model = model;
        this.transducer = transducer is null ? null : transducer with { Embeddings = transducer.Embeddings.ToArray() };
        var options = new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 };
        try
        {
            int channels = frontEnd == GraphFrontEnd.RawWaveform ? 1 : frontEnd == GraphFrontEnd.Whisper80 ? 80 : 128;
            for (int i = 0; i < stages.Length; i++)
            {
                var graph = RecursiveAI.OneModel.Model.Load(stages[i].Path, options, cancellationToken: cancellationToken);
                encoders.Add(graph);
                CheckContract(graph, 3, channels, 3, stages[i].Channels, transducer is null && i == stages.Length - 1);
                channels = stages[i].Channels;
            }
            if (transducer is not null)
            {
                predictor = RecursiveAI.OneModel.Model.Load(transducer.PredictorPath, options, cancellationToken: cancellationToken);
                joint = RecursiveAI.OneModel.Model.Load(transducer.JointPath, options, cancellationToken: cancellationToken);
                CheckContract(predictor, 2, transducer.StateSize, 2, transducer.StateSize, false);
                CheckContract(joint, 2, stages[^1].Channels + transducer.EmbeddingSize, 2, vocabulary.Pieces.Length + transducer.Durations.Length, false);
            }
        }
        catch { foreach (var graph in encoders) graph.Dispose(); predictor?.Dispose(); joint?.Dispose(); throw; }
    }

    /// <summary>Transcribes finite mono 16000 Hz PCM. Inputs shorter than 400 samples return no words; over 30 seconds use VAD-first chunks. Null or explicit language labels are retained without detection. Output times are clipped to the supplied audio horizon. Throws ArgumentException for audio shape/nonfinite values, InvalidModelShapeException for graph/decoder shape mismatches, ObjectDisposedException after disposal, and OperationCanceledException on cancellation.</summary>
    public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this); ArgumentNullException.ThrowIfNull(audio); cancellationToken.ThrowIfCancellationRequested();
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Graph recognition requires mono 16000 Hz PCM.", nameof(audio));
        foreach (float value in audio.Samples.Span) if (!float.IsFinite(value)) throw new ArgumentException("PCM must be finite.", nameof(audio));
        if (audio.FrameCount > 480000) { using var vad = new SileroVad(cancellationToken: cancellationToken); return new VadTranscriptionPipeline(this, vad).Transcribe(audio, language, cancellationToken); }
        if (audio.FrameCount < 400) return new() { Text = "", Language = language, Duration = audio.Duration, Models = [Model] };
        try
        {
            var (values, frames, channels) = Features(audio, cancellationToken);
            for (int i = 0; i < stages.Length; i++)
            {
                if (!stages[i].SamePadding && frames < stages[i].ReceptiveField) throw new InvalidModelShapeException("Input is shorter than the declared receptive field.");
                values = Run(encoders[i], values, [1, channels, frames], cancellationToken);
                frames = stages[i].SamePadding ? (frames + stages[i].Stride - 1) / stages[i].Stride : (frames - stages[i].ReceptiveField) / stages[i].Stride + 1;
                channels = stages[i].Channels;
                if (frames <= 0 || values.Length != (long)frames * channels) throw new InvalidModelShapeException("Encoder output does not match its declared channels and temporal stride.");
            }
            ImmutableArray<FrameToken> tokens;
            if (transducer is not null) tokens = DecodeTransducer(values, frames, channels, cancellationToken);
            else if (beam == 0) tokens = CtcGreedy.Decode(values, vocabulary.Pieces.Length, blank, cancellationToken);
            else
            {
                var hypothesis = CtcPrefixBeamSearch.Decode(values, vocabulary.Pieces.Length, blank, beam, cancellationToken: cancellationToken);
                // The trellis takes log probabilities, not raw logits.
                for (int frame = 0; frame < frames; frame++)
                {
                    int offset = frame * channels; float max = values.AsSpan(offset, channels).ToArray().Max(); double sum = 0;
                    for (int id = 0; id < channels; id++) sum += Math.Exp(values[offset + id] - max);
                    double norm = max + Math.Log(sum);
                    for (int id = 0; id < channels; id++) values[offset + id] -= (float)norm;
                }
                var alignment = CtcTrellis.Align(values, frames, channels, hypothesis.Tokens.AsSpan(), blank, cancellationToken: cancellationToken);
                if (!alignment.Succeeded) throw new InvalidModelShapeException("CTC beam hypothesis could not be aligned to its emissions.");
                tokens = alignment.Tokens.Select(t =>
                {
                    double probability = 0;
                    for (int f = t.StartFrame; f < t.EndFrame; f++) probability += Math.Exp(values[f * channels + t.TokenId]);
                    return new FrameToken(t.TokenId, t.StartFrame, t.EndFrame, (float)(probability / (t.EndFrame - t.StartFrame)));
                }).ToImmutableArray();
            }
            var decoded = vocabulary.Decode(tokens);
            var words = NativeWordTiming.Map(decoded.Timings, decoded.Words, frameDuration, cancellationToken: cancellationToken)
                .Select(w => w with { Start = w.Start > audio.Duration ? audio.Duration : w.Start, End = w.End > audio.Duration ? audio.Duration : w.End, Language = language }).ToImmutableArray();
            return new() { Text = decoded.Text, Words = words, Language = language, Duration = audio.Duration, Models = [Model], Segments = [new() { Text = decoded.Text, Words = words, Language = language, Start = TimeSpan.Zero, End = audio.Duration }] };
        }
        catch (ArgumentException) { throw new InvalidModelShapeException("Graph outputs must be finite, dimensionally valid logits and a nonoverlapping vocabulary word map."); }
        catch (SessionException) { throw new InvalidModelShapeException("A graph rejected the declared float32 input/output contract. Check tensor ranks, channel counts, predictor state and joint dimensions."); }
    }

    private (float[] Values, int Frames, int Channels) Features(AudioBuffer audio, CancellationToken ct)
    {
        if (frontEnd == GraphFrontEnd.RawWaveform) return (audio.Samples.ToArray(), audio.FrameCount, 1);
        int channels = frontEnd == GraphFrontEnd.Whisper80 ? 80 : 128;
        if (frontEnd != GraphFrontEnd.Nemo128) return (new WhisperLogMelSpectrogram(channels).Extract(audio, false, ct), audio.FrameCount / 160, channels);
        var padded = new NemoLogMelSpectrogram().Extract(audio, cancellationToken: ct); int frames = NemoLogMelSpectrogram.GetFrameCount(audio.FrameCount);
        var values = new float[128 * frames];
        for (int c = 0; c < 128; c++) padded.AsSpan(c * (frames + 1), frames).CopyTo(values.AsSpan(c * frames));
        return (values, frames, 128);
    }

    private ImmutableArray<FrameToken> DecodeTransducer(float[] encoded, int frames, int channels, CancellationToken ct)
    {
        var spec = transducer!; float[] state = Run(predictor!, new float[spec.StateSize], [1, spec.StateSize], ct);
        var input = new float[channels + spec.EmbeddingSize]; int accepted = 0;
        void Logits(int frame, ReadOnlySpan<int> prefix, Span<float> tokenLogits, Span<float> durationLogits, CancellationToken cancel)
        {
            while (accepted < prefix.Length)
            {
                if (state.Length != spec.StateSize) throw new InvalidModelShapeException("Predictor output state size differs from its declared size.");
                spec.Embeddings.Span.Slice(prefix[accepted++] * spec.EmbeddingSize, spec.EmbeddingSize).CopyTo(state);
                state = Run(predictor!, state, [1, spec.StateSize], cancel);
            }
            if (state.Length != spec.StateSize) throw new InvalidModelShapeException("Predictor output state size differs from its declared size.");
            for (int c = 0; c < channels; c++) input[c] = encoded[c * frames + frame];
            state.AsSpan(0, spec.EmbeddingSize).CopyTo(input.AsSpan(channels));
            float[] logits = Run(joint!, input, [1, input.Length], cancel);
            if (logits.Length != vocabulary.Pieces.Length + durationLogits.Length) throw new InvalidModelShapeException("Joint output must contain vocabulary logits followed by duration logits.");
            logits.AsSpan(0, tokenLogits.Length).CopyTo(tokenLogits); logits.AsSpan(tokenLogits.Length).CopyTo(durationLogits);
        }
        return spec.Durations.Length == 0
            ? RnntGreedy.Decode(frames, vocabulary.Pieces.Length, blank, Logits, spec.MaxSymbolsPerFrame, cancellationToken: ct)
            : TdtGreedy.Decode(frames, vocabulary.Pieces.Length, blank, spec.Durations.AsSpan(), Logits, spec.MaxSymbolsPerFrame, cancellationToken: ct);
    }
    private static void CheckContract(Model graph, int inputRank, int inputChannels, int outputRank, int outputChannels, bool frameMajor)
    {
        var inputs = graph.Session.Inputs; var outputs = graph.Session.Outputs;
        if (inputs.Length != 1 || outputs.Length != 1 || inputs[0].DType != DType.F32 || outputs[0].DType != DType.F32 || inputs[0].Rank != inputRank || outputs[0].Rank != outputRank)
            throw new InvalidModelShapeException("Graphs must have one float32 tensor input and output with the declared ranks.");
        bool Wrong(TensorInfo info, int axis, int expected) => info.Dims[axis].IsFixed && info.Dims[axis].Value != expected;
        if (Wrong(inputs[0], 0, 1) || Wrong(outputs[0], 0, 1) || Wrong(inputs[0], 1, inputChannels) || Wrong(outputs[0], frameMajor ? 2 : 1, outputChannels))
            throw new InvalidModelShapeException("Declared graph batch or channel dimensions differ from the supplied contract.");
    }
    private static float[] Run(Model model, float[] input, int[] shape, CancellationToken ct) { ct.ThrowIfCancellationRequested(); var result = model.Run(input, shape); ct.ThrowIfCancellationRequested(); return result; }
    /// <summary>Yields immutable VAD-first final segments. Input, cancellation, errors and language-label semantics match Transcribe.</summary>
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        foreach (var segment in new VadTranscriptionPipeline(this, vad).TranscribeStream(audio, language, cancellationToken)) yield return segment;
    }
    /// <summary>Always throws CapabilityNotSupportedException: no language-ID head is part of this contract. Cancellation is checked first.</summary>
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) { cancellationToken.ThrowIfCancellationRequested(); throw new CapabilityNotSupportedException("The custom graph contract does not include language identification."); }
    /// <summary>Releases all owned graphs; repeated disposal is harmless.</summary>
    public void Dispose() { if (disposed) return; disposed = true; foreach (var graph in encoders) graph.Dispose(); predictor?.Dispose(); joint?.Dispose(); }
}
