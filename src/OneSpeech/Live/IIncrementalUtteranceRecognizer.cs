namespace OneSpeech.Live;

// Session-local caches; models are borrowed. Returns only the immutable payload that will be emitted.
internal interface IIncrementalUtteranceRecognizer : IDisposable
{
    Segment Decode(ReadOnlyMemory<float> audio, string? language, bool final, TimeSpan start, TimeSpan end, CancellationToken cancellationToken);
    void Reset();
}
