# Third-party notices

## Model weights

No model weights are included.

## Data

The LibriSpeech clip identified below is the only third-party audio fixture. Other feature fixtures use deterministic synthetic signals.

## Libraries

Package dependencies retain their own licences and notices.

## Reference observations

The tone feature fixtures were produced from deterministic synthetic audio by black-box calls to openai-whisper 20250625 (MIT). They contain only numerical outputs, not model weights, source code or third-party audio. The oracle harness also supports transformers 5.22.2 (Apache-2.0). Neither oracle is a shipped dependency.

## LibriSpeech fixture

`testdata/1089-134686-0001.flac` is from the LibriSpeech ASR corpus (2015), Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. Test-clean utterance 1089-134686-0001 is distributed unmodified under Creative Commons Attribution 4.0: <https://creativecommons.org/licenses/by/4.0/>. Source: <https://www.openslr.org/12>. The archive path and SHA-256 are recorded in `testdata/manifest.json`.
