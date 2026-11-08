namespace OneSpeech.Live;

// Session-local caches; models are borrowed. Input is the growing retained utterance, not a new independent recording.
internal interface IIncrementalUtteranceRecognizer : IDisposable
{
    Transcript Decode(AudioBuffer audio, string? language, bool final, CancellationToken cancellationToken);
    void Reset();
}
