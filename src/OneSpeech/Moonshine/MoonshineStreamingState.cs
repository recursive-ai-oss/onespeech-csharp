using OneSpeech.Live;

namespace OneSpeech.Moonshine;

internal sealed class MoonshineStreamingState : IIncrementalUtteranceRecognizer
{
    private readonly MoonshineLiveWorkspace workspace;
    internal MoonshineStreamingState(MoonshineRecognizer recognizer) => workspace = new(recognizer);
    public Segment Decode(ReadOnlyMemory<float> audio, string? language, bool final, TimeSpan start, TimeSpan end, CancellationToken cancellationToken)
    {
        if (language is not (null or "en")) throw new CapabilityNotSupportedException("Moonshine streaming checkpoints support English only.");
        return workspace.Decode(audio, final, start, end, cancellationToken);
    }
    public void Reset() => workspace.Reset();
    public void Dispose() => workspace.Dispose();
}
