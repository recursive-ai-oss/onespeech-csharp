namespace OneSpeech.Live;

// Session-local caches; models are borrowed. Returns only the immutable payload that will be emitted.
internal interface IIncrementalUtteranceRecognizer : IDisposable
{
    // Preferred partial cadence on the mono 16 kHz sample clock; default 200 ms.
    int PartialCadenceSamples => 3200;
    // When not null, the utterance length (samples since its padded start) at which the next Decode can
    // produce new output. The session then decodes exactly then instead of on the fixed cadence, so a model
    // whose chunk completes at an offset from the cadence grid does not wait for the next cadence tick.
    int? NextOutputSamples => null;
    Segment Decode(ReadOnlyMemory<float> audio, string? language, bool final, TimeSpan start, TimeSpan end, CancellationToken cancellationToken);
    void Reset();
}
