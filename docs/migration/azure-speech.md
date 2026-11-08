# From Azure Speech SDK file recognition

Here “offline recognition” means recognizing a file rather than a microphone; it does **not** imply that an Azure service call itself is offline. This concept map uses Microsoft's [recognition guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech), consulted 2026-11-08. No SDK implementation is used.

| Familiar concept | OneSpeech equivalent |
|---|---|
| Speech configuration with service credentials/region | Local catalog selection; no cloud recognition credentials or region |
| Audio configuration selecting a WAV file | `Speech.Transcribe(path)`; supported audio is decoded through OneAudio |
| Single-utterance recognition | VAD-based recognition; a complete file call is not limited to one utterance |
| Continuous recognition callbacks | `await foreach` over file segments or live events |
| Recognizing / recognized notifications | Replacing `PartialEvent` / immutable `FinalEvent` |
| Serialized service result | `Transcript.ToJson()` with the versioned OneSpeech schema |

Do not copy service JSON parsing, event-threading assumptions or service offsets into a OneSpeech application. Every time here is relative to original input audio; confidence may be null and is not calibrated across models. Cancellation is explicit and output is an immutable document/event snapshot, not an SDK result object.

Recognition runs locally after models are obtained, but first use may access the network to download weights. For disconnected deployments, [pre-fetch all required roles](../models.md). Live input does not open a device. [Speaker recognition](../speaker-recognition.md) requires separate consent and legal consideration; a recognized voice is not an authentication guarantee.
