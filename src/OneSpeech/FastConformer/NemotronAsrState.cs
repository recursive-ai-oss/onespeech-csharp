using System.Collections.Immutable;
using OneSpeech.Audio;
using OneSpeech.Decoding;
using OneSpeech.Live;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.FastConformer;

// Per-utterance state. All graph buffers are caller-owned; emitted document snapshots are not reused.
internal sealed class NemotronAsrState : IIncrementalUtteranceRecognizer
{
    private readonly NemotronAsrRecognizer owner;
    private readonly NemoMelStream mel = new(new NemoLogMelSpectrogram());
    private readonly NemotronAsrEncoderState encoder;
    private readonly List<FrameToken> tokens = new(4096);
    private readonly Tensor promptInput, promptOutput, predictorInput, predictorOutput, jointInput, logits;
    private readonly Tensor[] promptInputs, promptOutputs, predictorInputs, predictorOutputs, jointInputs, jointOutputs;
    private readonly Action<float[]> acceptMel;
    private readonly bool padFinalChunk;
    private readonly bool automaticPrompt;
    private readonly float[] zeros = new float[160];
    private CancellationToken cancellation;
    private readonly PhraseTrie? phrases;
    private int samples, frames, promptId, phraseState;
    private bool initialized;
    internal string? DetectedLanguage { get; private set; }
    private static readonly string[] PromptInputs = ["prompt_input"], PromptOutputs = ["conditioned"], PredictorInputs = ["predictor_state"], PredictorOutputs = ["next_predictor_state"], JointInputs = ["joint_input"], JointOutputs = ["logits"];

    internal NemotronAsrState(NemotronAsrRecognizer owner, int promptId, CancellationToken ct, bool padFinalChunk = true, PhraseTrie? phrases = null)
    {
        this.owner = owner; this.phrases = phrases; this.promptId = promptId; this.padFinalChunk = padFinalChunk; automaticPrompt = promptId == 101;
        int chunk = owner.ChunkFrames;
        promptInput = Tensor.Zeros(DType.F32, Shape.Create(1, 1152, chunk)); promptOutput = Tensor.Zeros(DType.F32, Shape.Create(1, 1024, chunk));
        predictorInput = Tensor.Zeros(DType.F32, Shape.Create(1, 3200)); predictorOutput = Tensor.Zeros(DType.F32, Shape.Create(1, 3200));
        jointInput = Tensor.Zeros(DType.F32, Shape.Create(1, 1664)); logits = Tensor.Zeros(DType.F32, Shape.Create(1, 13088));
        promptInputs = [promptInput]; promptOutputs = [promptOutput]; predictorInputs = [predictorInput]; predictorOutputs = [predictorOutput]; jointInputs = [jointInput]; jointOutputs = [logits];
        acceptMel = Accept;
        NemotronAsrEncoderState? created = null;
        try
        {
            created = new(owner.First, owner.Last, owner.Tail, Catalog.Conversion.NemotronAsrConverter.Dimensions, chunk, Encoded, ct);
            encoder = created; Initialize(ct);
        }
        catch
        {
            created?.Dispose(); promptInput.Dispose(); promptOutput.Dispose(); predictorInput.Dispose(); predictorOutput.Dispose(); jointInput.Dispose(); logits.Dispose();
            throw;
        }
    }
    private void Accept(float[] values)
    {
        foreach (float value in values) if (!float.IsFinite(value)) throw new UnsupportedAudioException("PCM magnitude exceeds the finite feature range; normalize the input level.");
        encoder.Append(values, cancellation);
    }
    private void Initialize(CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        owner.Embeddings.AsSpan(13087 * 640, 640).CopyTo(predictorInput.MutableData<float>());
        owner.Predictor.Run(PredictorInputs, predictorInputs, PredictorOutputs, predictorOutputs);
        initialized = true;
    }
    private void Encoded(ReadOnlySpan<float> values, int valid, CancellationToken ct)
    {
        int chunk = owner.ChunkFrames;
        values.CopyTo(promptInput.MutableData<float>());
        promptInput.MutableData<float>().Slice(1024 * chunk).Clear();
        promptInput.MutableData<float>().Slice((1024 + promptId) * chunk, chunk).Fill(1);
        owner.Prompt.Run(PromptInputs, promptInputs, PromptOutputs, promptOutputs);
        for (int frame = 0; frame < valid; frame++, frames++)
        {
            ct.ThrowIfCancellationRequested();
            for (int c = 0; c < 1024; c++) jointInput.MutableData<float>()[c] = promptOutput.Data<float>()[c * chunk + frame];
            for (int symbol = 0; symbol < 10; symbol++)
            {
                predictorOutput.Data<float>()[..640].CopyTo(jointInput.MutableData<float>()[1024..]);
                owner.Joint.Run(JointInputs, jointInputs, JointOutputs, jointOutputs);
                var scores = logits.Data<float>(); int best = 0;
                for (int i = 1; i < scores.Length; i++) if (scores[i] > scores[best]) best = i;
                if (phrases is not null)
                {
                    // Shallow fusion: the phrase bonus is added to token log-scores before the choice; blank gets none.
                    // Same rule as ContextualTransducerSearch (Le et al. 2021 §3.1), kept token-for-token identical.
                    double top = double.NegativeInfinity;
                    for (int i = 0; i < scores.Length; i++)
                    {
                        double score = scores[i] + (i == 13087 ? 0 : phrases.Advance(phraseState, i, out _));
                        if (score > top) { top = score; best = i; }
                    }
                }
                if (best == 13087) break;
                if (phrases is not null) { phrases.Advance(phraseState, best, out int next); phraseState = next; }
                if (tokens.Count >= 40000) throw new CapabilityNotSupportedException("Nemotron utterances are limited to 40000 tokens; split the audio.");
                double sum = 0; for (int i = 0; i < scores.Length; i++) sum += Math.Exp(scores[i] - scores[best]);
                DetectedLanguage ??= owner.Vocabulary.Language(best);
                if (!owner.Vocabulary.IsSpecial(best)) tokens.Add(new(best, frames, frames + 1, (float)(1 / sum)));
                predictorOutput.Data<float>().CopyTo(predictorInput.MutableData<float>());
                owner.Embeddings.AsSpan(best * 640, 640).CopyTo(predictorInput.MutableData<float>());
                owner.Predictor.Run(PredictorInputs, predictorInputs, PredictorOutputs, predictorOutputs);
            }
        }
    }
    public int PartialCadenceSamples => owner.ChunkFrames * 1280;
    // The encoder emits a chunk as soon as its last mel frame exists. Mel frame f needs 160 f + 200 samples (400-sample
    // Hann support), the first chunk holds 8c - 7 frames and later chunks 8c, so chunk j completes at 1280 c (j + 1) - 1080.
    public int? NextOutputSamples { get { int grid = owner.ChunkFrames * 1280; return grid * ((samples + 1080) / grid + 1) - 1080; } }
    internal int TokenCount => tokens.Count;
    internal ImmutableArray<FrameToken> Tokens => [.. tokens];
    internal void Push(ReadOnlySpan<float> audio, bool final, CancellationToken ct)
    {
        cancellation = ct; if (!initialized) Initialize(ct);
        samples += audio.Length;
        if (final && padFinalChunk && samples > 0)
        {
            mel.Push(audio, false, acceptMel, ct);
            // The released streaming generate API requires a full steady chunk at EOF.
            // Centered FFT context begins 256 samples before the next mel center; zeros
            // extend that context without extending the caller's real audio-time clock.
            int first = owner.ChunkFrames * 8 - 7, steady = owner.ChunkFrames * 8;
            int remaining = samples - first * 160 + 256;
            int count = Math.Max(0, (remaining + steady * 160 - 1) / (steady * 160));
            int targetFrames = first + count * steady;
            int paddedSamples = (targetFrames - 1) * 160 + 200;
            for (int extra = Math.Max(0, paddedSamples - samples); extra > 0;)
            {
                int length = Math.Min(extra, zeros.Length); mel.Push(zeros.AsSpan(0, length), false, acceptMel, ct); extra -= length;
            }
            mel.Push([], true, acceptMel, ct); encoder.Complete(ct);
        }
        else
        {
            mel.Push(audio, final, acceptMel, ct);
            if (final) encoder.Complete(ct);
        }
    }
    public Segment Decode(ReadOnlyMemory<float> audio, string? language, bool final, TimeSpan start, TimeSpan end, CancellationToken cancellationToken)
    {
        promptId = automaticPrompt ? 101 : owner.PromptId(language);
        if (audio.Length < samples) throw new InvalidOperationException("Utterance audio must grow monotonically.");
        Push(audio.Span[samples..], final, cancellationToken);
        return owner.Snapshot(Tokens, start, end, automaticPrompt ? DetectedLanguage : language ?? DetectedLanguage, cancellationToken);
    }
    public void Reset()
    {
        encoder.Reset(); mel.Reset(); tokens.Clear(); samples = frames = phraseState = 0; initialized = false; DetectedLanguage = null;
        predictorInput.MutableData<float>().Clear(); predictorOutput.MutableData<float>().Clear();
    }
    public void Dispose()
    {
        encoder.Dispose(); promptInput.Dispose(); promptOutput.Dispose(); predictorInput.Dispose(); predictorOutput.Dispose(); jointInput.Dispose(); logits.Dispose();
    }
}
