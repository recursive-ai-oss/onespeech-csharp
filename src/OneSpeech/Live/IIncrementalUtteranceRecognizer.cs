namespace OneSpeech.Live;

// Session-local caches; models are borrowed. Returns only the immutable payload that will be emitted.
internal interface IIncrementalUtteranceRecognizer : IDisposable
{
    // Preferred partial cadence on the mono 16 kHz sample clock; default 200 ms.
    int PartialCadenceSamples => 3200;
    Segment Decode(ReadOnlyMemory<float> audio, string? language, bool final, TimeSpan start, TimeSpan end, CancellationToken cancellationToken);
    void Reset();
}
