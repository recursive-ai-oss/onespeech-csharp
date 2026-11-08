# Contextual phrase biasing

Contextual biasing favours domain names and phrases during recognition; it is not model training and cannot guarantee that a phrase is spoken. The low-level entry point is `SpeechRecognizer.WithPhrases`. Check the recognizer's `Capabilities.Biasing` before requesting it. Unsupported requests must fail with a typed capability error rather than silently ignoring the phrases.

The family determines how context is applied: Whisper prompt conditioning, Qwen context, or CTC/transducer shallow fusion over a phrase trie. Phrase weighting is not a probability and does not change the published checkpoint. Overweighting names can insert them into unrelated speech; compare both phrase recall and ordinary WER on held-out audio.

The phrase-binding API and cross-family results are being finalized. This page intentionally provides no call-signature example until that API is present in the shipped build. The current checkout's base recognizer has no `WithPhrases` member; do not assume an older package supports it. File helper parameters remain only language and model; use the low-level recognizer for contextual options.
