# Speaker recognition

Diarization asks **who spoke when**, using anonymous labels. Recognition asks whether speech matches an enrolled identity. `InMemorySpeakerRegistry` borrows a `SpeakerEmbedder`, holds enrollment vectors only in memory and clears them on disposal. Keep the embedder alive until the registry is disposed. It does not persist a voice database.

The following method accepts an already loaded embedder and consented mono PCM at its required sample rate. Choose the identification threshold on disjoint development trials; 0.3 here is illustrative, not a universal biometric operating point.

```csharp
static void RecognizeSpeaker(OneSpeech.SpeakerEmbedder embedder,
    OneSpeech.AudioBuffer enrollment, OneSpeech.AudioBuffer query,
    CancellationToken cancellationToken)
{
    using var registry = new OneSpeech.Speakers.InMemorySpeakerRegistry(embedder, threshold: 0.3);
    registry.Enroll("alice", enrollment, cancellationToken);
    var match = registry.Identify(query, cancellationToken);
    double score = registry.Verify(enrollment, query, cancellationToken);
    Console.WriteLine(match);
    Console.WriteLine(score);
}
```

Enrollment replaces the named identity. Multiple utterances are normalized and averaged with equal weight. Identification may return unknown below the threshold; verification returns a similarity score, **not a probability or calibrated identity decision**. See [embedding and scoring details](speakers.md), and [results](results.md#speaker-embeddings) for the limited trial population. File-based `enroll`, `identify` and `verify` are available in the [CLI](cli-and-model-packages.md); `Speech` does not yet expose these as statics.

Embeddings are biometric data. Obtain informed consent and check local law, including Illinois BIPA and GDPR Article 9. Apply appropriate access controls and deletion policies to enrollment audio, transcripts and application logs. Not persisting embeddings in this library does not remove the host's legal obligations. Never identify a person merely from a document-local diarization label. This is not legal advice.
