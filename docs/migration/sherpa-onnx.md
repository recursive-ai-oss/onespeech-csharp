# From sherpa-onnx .NET

This guide maps concepts from the [published C# documentation](https://k2-fsa.github.io/sherpa/onnx/csharp-api/index.html), consulted 2026-11-08. It is not a shim and does not import model configuration objects or native handles.

| Familiar concept | OneSpeech equivalent |
|---|---|
| Offline recognition of a waveform | `Speech.Transcribe` for a file, or `SpeechRecognizer.Transcribe` for PCM |
| Streaming recognition | `LiveSession` or `Speech.Listen` over caller-supplied PCM |
| Separately selected recognition and speaker models | Role interfaces plus typed family loaders and `ModelInfo` provenance |
| Custom exported CTC/transducer models | `GraphSpeechRecognizer` with an explicitly declared tensor/feature contract |
| Speaker diarization output | `Turn` values and word assignments in a `Transcript` |

An arbitrary ONNX export is not automatically compatible. OneSpeech runs supported graphs on OneModel, not ONNX Runtime. Confirm layout, blank ID, front end, temporal stride and predictor/joint state contract in [custom speech graphs](../custom-models.md). There is no native-handle escape hatch.

Live input is mono 16 kHz PCM; capture and resampling remain the caller's responsibility. `PartialEvent` replaces a hypothesis for one utterance; `FinalEvent` commits it. Do not infer finality from an unchanged partial. See [the event model](../live-sessions.md) and [diarization limitations](../diarization.md).
