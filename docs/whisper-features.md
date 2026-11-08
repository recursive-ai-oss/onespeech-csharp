# Whisper features

`OneSpeech.Audio.WhisperLogMelSpectrogram` consumes at most 480,000 samples of 16 kHz mono float PCM. Resampling and channel mixing belong to the audio input layer. Samples are not clipped. Nonfinite samples and signals whose power exceeds float32 range are rejected.

The default 80-bin variant covers tiny through large-v2. Large-v3 and large-v3-turbo use 128 bins, as specified by their published `preprocessor_config.json`. Do not infer the bin count from model size alone.

## Transform

- Periodic Hann, 400 samples (25 ms), hop 160 samples (10 ms).
- Zero-pad short signals to 480,000 samples **before** centered reflection padding of 200 samples at each side. Reflection excludes the boundary sample itself. An empty padded signal is silence.
- Unnormalized forward Fourier transform, squared complex magnitudes at 201 nonnegative frequency bins. Drop the final STFT frame.
- Slaney's piecewise mel scale: linear below 1 kHz with spacing 200/3 Hz per mel unit; logarithmic above it, with 27 units from 1 to 6.4 kHz. Equally spaced mel edges from 0 to 8 kHz. Triangular filters, area-normalized by 2/(right edge − left edge) in Hz.
- Floor mel power at 1e-10, take log10, floor again at the maximum log power over the entire window minus 8. Return `(log power + 4) / 4`.
- Output is a flat mel-major float array: element `(mel, frame)` is `mel * frames + frame`. Padded windows always have 3000 frames.

`padToWindow: false` supports feature probes with their natural length, not model inference: frame count is `floor(samples / 160)`. Input must exceed 200 samples so centered reflection is defined. Longer files must be windowed by the caller; this extractor neither silently truncates nor allocates in proportion to file length. Cancellation is checked at every frame.

The algorithm follows Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022), §2.2. Padding, normalization and edge behavior were confirmed with black-box calls to the feature APIs below, not implementation source.

## Numerical checks

The feature acceptance tolerance is maximum absolute error **3e-5**, mean absolute error **1e-6** against float32 CPU feature outputs. This is a feature tolerance, not OneModel's graph exact-mode contract. The managed mixed-radix Fourier transform uses double complex accumulation, then float32 powers and mel sums; reference FFT and matrix-reduction orders need not be bit-identical.

`tools/oracles/whisper-features/features.py` calls only the public `whisper.log_mel_spectrogram` API (openai-whisper 20250625) and `WhisperFeatureExtractor` (transformers 5.22.2). Its locked environment pins torch 2.15.0 CPU, numpy 2.5.4 and soundfile 0.14.0. `prepare` decodes four LibriSpeech test-clean clips to shared float arrays and generates deterministic synthetic inputs. `whisper` and `transformers` write reference features; `tools/OneSpeech.WhisperProbe` compares the managed extractor to both. Model weights are not required.

Example, from the repository root, with the corpus and scratch paths supplied by the caller:

```sh
export UV_PROJECT_ENVIRONMENT="${ORACLE_ENV:?external environment directory required}"
uv sync --project tools/oracles/whisper-features --locked
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
$PY tools/oracles/whisper-features/features.py prepare "$OUTPUT" --librispeech "$TEST_CLEAN"
$PY tools/oracles/whisper-features/features.py whisper "$OUTPUT"
$PY tools/oracles/whisper-features/features.py transformers "$OUTPUT"
dotnet run --project tools/OneSpeech.WhisperProbe -- "$OUTPUT"
```

Use `UV_PROJECT_ENVIRONMENT` to put the environment outside the checkout. Inputs and feature outputs also stay outside the checkout. The preparation manifest records clip IDs and hashes. It chooses test-clean IDs 1089-134686-0000 through 1089-134686-0003 in sorted order, plus a 440 Hz tone, empty input, a single-sample impulse, exactly 400 samples, DC and alternating clipped samples. No evaluation audio is committed. Small tone-value fixtures in the tests record observed output of openai-whisper 20250625.

## Graph conversion and recognition

`tools/OneSpeech.Convert fetch <tiny|small|large-v3-turbo> <directory>` fetches the explicitly pinned upstream data and verifies the safetensors SHA-256. `whisper <directory>` verifies the weight digest again and writes independently constructed encoder and explicit-cache decoder graphs in OneModel format. The encoder also computes cross-attention keys and values once per window. Because OneModel 1.0.0 limits each model file to 2 GiB minus one byte, turbo's encoder is partitioned after its first 16 blocks into `encoder-part0.onemodel` and `encoder.onemodel`.

`WhisperRecognizer.Load(directory, options)` loads local graphs and data without network access. Its offline path uses timestamp tokens to advance through 30-second windows, bounded previous-text conditioning, per-window cancellation and optional audio-time progress. The default temperature schedule is 0/.2/.4/.6/.8/1, with compression ratio 2.4, average log probability -1 and no-speech probability .6. Temperature zero is greedy, not the CLI's beam-search configuration. Positive temperatures sample one candidate with an explicit seed. Native word timestamps are not advertised; optional cross-attention DTW supplies word timing provenance.

The graph and recognition paths are still under numerical and reference validation. In particular, encoder scaled error, special-token decoding and DTW boundary behavior are not yet accepted; feature parity alone must not be read as end-to-end model parity. Reference inference and evaluation drivers under `tools/oracles/whisper-features` write per-utterance files for resumption rather than hiding incomplete runs. No quality claim is made here.
