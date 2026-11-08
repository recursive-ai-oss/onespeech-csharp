# Speech recognition for .NET

OneSpeech is building transcription, word timestamps, diarization, speaker recognition and live sessions in managed .NET, without native code. Open-weight models run on OneModel, with audio input through OneAudio.

**Status: in development, no public release yet.** Whisper and Qwen3-ASR recognition, forced alignment, VAD and speaker processing run natively on OneModel. Shared contracts and bounded transcript JSON remain the document seam.

OneSpeech is unaffiliated with other products or companies named OneSpeech.

Targets .NET 10. Licensed under MIT OR Apache-2.0; see LICENSE-MIT and LICENSE-APACHE.

## Development

Set `ONESPEECH_ARTIFACTS` to an absolute directory outside the checkout, then run `dotnet restore`, `dotnet build -c Release --no-restore`, and `dotnet test tests/OneSpeech.Tests -c Release --no-build`. The default output directory is `onespeech-build` in the system temporary directory. `dotnet pack -c Release --no-build` creates local packages only.

See [Qwen3 recognition and alignment](docs/qwen3.md), [the transcript JSON format](docs/transcript-json.md) and [the shared contracts](docs/contracts.md).

## Containers and captions

`RecursiveAI.OneSpeech.Media` adds OneMedia container audio input, subtitle export and MP4 caption muxing. Call `OneSpeech.Media.MediaInput.Register()` once at startup; then `Speech.Transcribe("meeting.mkv")` uses the container's audio track. Transcript extensions `ToSrt`, `ToVtt`, `ToAss` and `ToTtml` use OneMedia's writers. See [container audio and captions](docs/media.md) for untimed words, fidelity reports, streams and muxing.

## Contextual biasing

Biasing is a low-level request; `Speech.*` stays unchanged. A loaded recognizer
returns a borrowed wrapper with its own immutable phrase list:

```csharp
using var model = OneSpeech.Whisper.WhisperRecognizer.Load(modelDirectory);
SpeechRecognizer biased = model.WithPhrases([new Phrase("Acme Photonics"), new Phrase("Marisol")]);
Transcript result = biased.Transcribe(audio, "en", cancellationToken);
// The same biased recognizer can be passed to VadTranscriptionPipeline.
```

Keep the owning model alive until the wrappers and their sessions are finished.
Wrappers do not own or dispose it; the model's concurrency limits still apply.
Calling `WithPhrases` again replaces the list; an empty list returns the original
unbiased recognizer. The original recognizer is never modified.

Whisper conditions each window with ordinary prompt tokens. Qwen3-ASR appends
phrases to its existing system context. These prompt methods refuse explicit
numeric weights rather than pretending to apply a score bonus. Parakeet TDT
uses token-level shallow fusion; `new Phrase("Acme", 0.5)` supplies a natural-log
bonus per token (null defaults to 1). Custom CTC/transducer graphs use the same
fusion after the caller supplies `TokenVocabulary.WithPhraseEncoder` for their
exact tokenizer. CTC biasing uses prefix beam search, even if the unbiased
recognizer is greedy. Moonshine does not support biasing: a nonempty request
throws `CapabilityNotSupportedException`, as declared by `Capabilities.Biasing`.

Biasing is not a spelling replacement or a guarantee that a term will be spoken.
It can introduce false alarms. Check both phrase recall and overall WER, with
absent-phrase controls, on your audio. See [contextual scoring and bounds](docs/contextual-biasing.md)
for prompt budgets, weights, decoder semantics and the repeatable evaluation.
