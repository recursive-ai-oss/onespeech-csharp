# Speaker recognition

Diarization asks **who spoke when**, using anonymous labels. Recognition asks whether speech matches an enrolled identity. `InMemorySpeakerRegistry` borrows a `SpeakerEmbedder`, holds enrollment vectors only in memory and clears them on disposal. Keep the embedder alive until the registry is disposed. It does not persist a voice database.

## File and stream helpers

```csharp
OneSpeech.Speech.Enroll("alice", "enrollment.wav");
OneSpeech.SpeakerMatch match = OneSpeech.Speech.Identify("query.wav");
double cosine = OneSpeech.Speech.Verify("first.wav", "second.wav");
OneSpeech.Speech.ClearSpeakers();
```

`SpeakerMatch.SpeakerId` is the best enrolled name, or null for unknown. `Score` is raw cosine, not a probability; an empty registry returns negative infinity. `Verify` scores two samples without enrollment and does not make a security decision. Enrollment replaces a case-sensitive name only after successful extraction. All helpers have explicit `CancellationToken` overloads. The only optional selectors are `language` and `model`: English CAM++ by default, Chinese CAM++ for `zh`, `zh-*` or `Chinese`, or either embedder's catalog ID explicitly. Enrollment spaces are separate per model. Other model IDs are refused with `CapabilityNotSupportedException`.

Samples are downmixed and resampled to mono 16 kHz, must contain at least 45 ms and at most 60 seconds, and should contain one consenting speaker's speech. These helpers do not automatically select speech or separate multiple speakers. Silent or extremely short samples cannot establish a reliable identity. Missing models download on first use; loaded speaker models remain in memory until `ClearSpeakers`. Operations are serialized so replacement and clearing are atomic with respect to speaker queries. Clearing is best effort, not a guarantee against memory dumps or runtime copies.

`EnrollAsync(name, stream)`, `IdentifyAsync(stream)` and `VerifyAsync(firstStream, secondStream)` consume readable encoded-audio streams from their current positions and leave them open. Encoded input is temporarily spooled to disk for decoding and deleted on success, cancellation or error. Enrollment embeddings are never persisted. CPU model execution stays synchronous; only stream and model I/O is awaited.

`Speech.Transcribe` / `TranscribeAsync` identify each diarization cluster's normalized mean window embedding against compatible enrollments. Accepted names populate `SpeakerLabel.Name` on turns and assigned words. IDs remain `speaker-1`, etc.; unassigned words and rejected clusters are unchanged. Different clusters may match the same name; this does not merge clusters or alter timings. Streaming segment and live event helpers do not retrospectively name speakers. A later clear or reenrollment never mutates a returned transcript.

## Low-level registry

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

Enrollment replaces the named identity. Multiple utterances are normalized and averaged with equal weight. Identification may return unknown below the threshold; verification returns a similarity score, **not a probability or calibrated identity decision**. See [embedding and scoring details](speakers.md), and [results](results.md#speaker-embeddings) for the limited trial population. File-based `enroll`, `identify` and `verify` are also available in the [CLI](cli-and-model-packages.md). The low-level registry retains explicit caller-controlled thresholds and multi-utterance enrollment.

Embeddings are biometric data. Obtain informed consent and check local law, including Illinois BIPA and GDPR Article 9. Apply appropriate access controls and deletion policies to enrollment audio, transcripts and application logs. Not persisting embeddings in this library does not remove the host's legal obligations. Never identify a person merely from a document-local diarization label. This is not legal advice.
