# Contextual phrase biasing

Contextual biasing favours domain names and phrases during recognition; it is not model training and cannot guarantee that a phrase is spoken. The low-level entry point is `SpeechRecognizer.WithPhrases`. Check the recognizer's `Capabilities.Biasing` before requesting it. Unsupported requests must fail with a typed capability error rather than silently ignoring the phrases.

The family determines how context is applied: Whisper prompt conditioning, Qwen context, or CTC/transducer shallow fusion over a phrase trie. Phrase weighting is not a probability and does not change the published checkpoint. Overweighting names can insert them into unrelated speech; compare both phrase recall and ordinary WER on held-out audio.

`SpeechRecognizer.WithPhrases(phrases)` is available since 1.0.0. It returns a biased wrapper and leaves the original recognizer unchanged. Whisper, Qwen3-ASR, Parakeet, Nemotron 3.5 ASR and custom CTC/transducer graphs support it; Moonshine refuses with `CapabilityNotSupportedException`. See the [README example](../README.md) for a call, and [results](results.md) for measured recall and WER per family. File helper parameters remain only language and model; use the low-level recognizer for contextual options.
