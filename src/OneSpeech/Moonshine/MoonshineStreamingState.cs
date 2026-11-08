using OneSpeech.Live;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Moonshine;

// The encoder has a finite receptive field and no positional embeddings (Streaming paper §3.2).
// Re-encode only the bounded suffix; frozen prefix states are retained, not recomputed with the utterance.
internal sealed class MoonshineStreamingState : IIncrementalUtteranceRecognizer
{
    private readonly MoonshineRecognizer recognizer;
    private readonly float[] prefix;
    private int frozenFrames;
    private int previousFrames;
    internal MoonshineStreamingState(MoonshineRecognizer recognizer)
    {
        this.recognizer = recognizer;
        prefix = new float[4096 * recognizer.Runtime.Configuration.EncoderWidth];
    }
    public Transcript Decode(AudioBuffer audio, string? language, bool final, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (language is not (null or "en")) throw new CapabilityNotSupportedException("Moonshine streaming checkpoints support English only.");
        var runtime = recognizer.Runtime;
        var c = runtime.Configuration;
        int frames = ((audio.FrameCount + 79) / 80 + 3) / 4;
        if (frames == 0) return new Transcript { Text = "", Duration = audio.Duration, Language = "en", Models = [recognizer.Model] };
        if (frames > 4096) throw new UnsupportedAudioException("A Moonshine streaming utterance cannot exceed 4096 encoder frames; use bounded VAD endpointing.");
        if (frames < previousFrames) frozenFrames = 0;
        previousFrames = frames;
        // Four layers have three frames of right context. Each layer has 15 past frames;
        // the two causal frontend convolutions add three encoder frames of left context.
        int begin = Math.Max(0, frozenFrames - (c.EncoderLayers * 15 + 3));
        using var suffix = runtime.Encode(new AudioBuffer(audio.Samples[(begin * 320)..], 16000), cancellationToken);
        suffix["encoder"].Data<float>()[((frozenFrames - begin) * c.EncoderWidth)..].CopyTo(prefix.AsSpan(frozenFrames * c.EncoderWidth));
        frozenFrames = final ? frames : Math.Max(frozenFrames, frames - 12);
        using var encoded = Tensor.FromSpan<float>(prefix.AsSpan(0, frames * c.EncoderWidth), Shape.Create(1, frames, c.EncoderWidth));
        return recognizer.DecodeEncoded(encoded, audio.Duration, cancellationToken);
    }
    public void Reset() { frozenFrames = 0; previousFrames = 0; }
    public void Dispose() { Array.Clear(prefix); frozenFrames = 0; }
}
