# NeMo features and transducer decoding

## Features

`NemoLogMelSpectrogram` consumes mono 16 kHz float PCM. It implements the
Parakeet v3 preprocessor parameters: 128 Slaney-area-normalized mel bands,
400-sample symmetric Hann window centered in a 512-point FFT, 160-sample hop,
preemphasis 0.97, power spectrum and natural log with additive `2^-24` guard.
80 mel bands are also supported. Dithering and augmentation are disabled.
Input is limited to 320–960,000 samples; callers must resample, downmix and
chunk outside this extractor.

Centered STFT uses **zero** padding. This edge behavior was confirmed against
NeMo Toolkit 3.0.0's public preprocessor API, not inferred from other STFT
implementations. The valid frame count is `floor(sample_count / 160)`. The
physical time axis has one extra centered frame, masked to zero; optional
`padTo` rounds that physical axis up to a multiple. The default zero disables
rounding, matching the Parakeet v3 model configuration. Padding follows
normalization, never participates in statistics, and is zero.

Per-feature normalization uses sample variance (denominator `valid_frames-1`)
and adds `1e-5` to the standard deviation. Float32 reduction roundoff can
produce a small nonzero normalized value even on digital silence. The
implementation retains float32 mean arithmetic rather than forcing silent
channels to zero. Extraction checks cancellation per STFT frame and channel;
instances are immutable and may be shared between threads.

Correctness probes use the first five lexicographically sorted LibriSpeech
test-clean clips. Against NeMo Toolkit 3.0.0, Torch 2.11.0 CPU, with dithering
zero, the maximum normalized absolute error is below `1e-4`; raw log-energy
error is below `1.1e-4`. Synthetic silence and 400/512-sample edges also meet
`1e-4` normalized error. Constant DC, whose variance is nearly zero in some
channels, uses `1e-3` normalized tolerance (`1.1e-4` before normalization).
These are float32 tolerances, not an exact-mode or whole-corpus parity claim.
`tools/oracles/nemo` records the released oracle pins and creates external
fixtures; `OneSpeech.NemoProbe` checks their dimensions and tolerances.

## Decoding

- `CtcGreedy` picks the smallest token ID on ties, merges consecutive equal
  IDs, then removes blanks. A blank separates repeated tokens. Token intervals
  cover the complete winning frame run; confidence is the mean conditional
  token probability over that run.
- `CtcPrefixBeamSearch` tracks distinct blank/nonblank probability sums for
  each prefix and merges paths before pruning. Scores are natural log
  probabilities without length normalization, language models or phrase bias.
  Ties use lexicographic token order. The result has no inferred timing.
- `RnntGreedy` follows Xu et al. 2023 Algorithm 1: nonblank updates the predictor
  prefix without advancing acoustic time; blank advances one frame. Repeated
  nonblanks are retained. Its reported token interval is one encoder frame,
  not a measured phonetic duration.
- `TdtGreedy` follows Algorithm 2: token and duration heads have independent
  argmax selections. Nonblank updates the prefix, and the selected duration
  advances acoustic time. Zero-duration tokens retain zero-length intervals;
  no minimum word duration is invented. A blank with zero duration advances
  one frame defensively, preventing an infinite loop. Positive duration
  intervals are clipped to the encoder-frame count.

Transducer callbacks receive the complete accepted nonblank prefix and must
fill both output spans; they may cache predictor state by prefix length.
NaN, positive infinity, all-masked rows and incomplete callback outputs are
errors. Symbol/token bounds throw instead of silently losing tokens.
All decoding is synchronous, deterministic and cancellation-aware.

`NativeWordTiming.Map` takes explicit tokenizer-produced word/token ranges.
It unions only those token intervals, averages token confidences, applies the
encoder-frame duration and chunk offset, and marks timed words `Native`.
Zero-token words remain untimed. It does not split text heuristically, fill
blank gaps, estimate missing boundaries, or assign speakers. At 8x encoder
subsampling, one 10 ms mel hop corresponds to 80 ms of audio time.

## Graph construction

The internal FastConformer graph builder emits OneModel-format graphs from an
abstract float tensor source. Its canonical tensor names are a **converter
schema**, not publisher tensor names. A checkpoint-specific mapping and
numerical parity are required before connecting it to a released model.
`ParakeetTensorSource` maps the pinned publisher header into this schema.

The encoder supports 8x depthwise-separable subsampling, relative-position
multihead attention, the Conformer convolution module and two half-step
feed-forwards. Single-batch graphs have no padding within the valid time axis.
They can be split at block boundaries, with each file required to stay below
2 GiB. Weights are copied at save time rather than materialized together.
Predictor graphs use an explicit packed input/output layout:
`[embedding_or_predictor_vector, h0, c0, h1, c1, ...]`. Each LSTM layer uses
input/forget/candidate/output gate order. Joint graphs accept concatenated
encoder and predictor vectors and return token logits (including blank),
followed by duration logits; zero duration columns selects the RNN-T shape.

Small deterministic random-weight tests exercise encoder, split-vs-unsplit,
predictor and joint graph loading/running on OneModel. Publisher-weight encoder
probes use the same five oracle feature arrays as the feature probes. Their
fp32 maximum absolute tolerance is 2e-5 (fast arithmetic, not bitwise exact
mode). Subsampling kernels operate on time then frequency. The masked final
STFT frame is cropped before subsampling; retaining it changes edge outputs
on odd frame lengths, as observed through NeMo's public encoder API.

## Parakeet-TDT-v3 recognition

Select this model explicitly:

```csharp
var transcript = Speech.Transcribe("recording.flac", model: "nvidia/parakeet-tdt-0.6b-v3");
```

The checkpoint is CC-BY-4.0, ungated **tier 2**: never a default, a package or a
mirror asset. First use fetches the pinned publisher safetensors and tokenizer,
verifies SHA-256, then converts locally into the per-user cache. The two
12-layer encoder graphs are each below 2 GiB, so OneModel 1.0.0 can load them
by path. Predictor, joint and embedding artifacts are local outputs too.
Their sizes and digests are catalog metadata, not download routes. Nothing
converted is redistributed. `ModelInfo` retains NVIDIA attribution, licence,
publisher revision, upstream digests and local output digests; format conversion
is marked and weights are unchanged.

`ParakeetRecognizer.Load` is the lower-level CPU entry point for a verified
`ResolvedModel`. It accepts mono 16 kHz PCM, uses greedy independent token and
duration argmax with durations [0,1,2,3,4], blank ID 8192, and a 10-symbol bound
per encoder frame. The initial predictor input is zero with zero LSTM state;
accepted tokens update the predictor exactly once. Its state is local to each
chunk. There is no beam, contextual biasing or translation in this adapter.
The output vocabulary uses the publisher's Metaspace decoder: leading ▁ starts
a word and punctuation remains attached to its preceding token range. Native
80 ms token-duration intervals are unioned into word intervals; zero durations
remain zero and final intervals are clipped to the input audio horizon.
Punctuation-only emissions retain their text and acoustic-clock advance but
contribute a zero-duration token timing interval. Word endpoints and confidence
stop at the last lexical emission, excluding trailing punctuation even when
it is emitted after a blank gap. This matches NeMo's observed timestamp output. The tokenizer's unknown surface is spaced U+2047, not the literal
`<unk>` vocabulary spelling; this too is checked through the public oracle.

Input above 30 seconds and all file helpers use VAD-first bounded chunks. Shorter
than 20 ms produces no words. Immutable final segments share one audio clock.
A supplied supported language labels the document but does not condition this
automatic multilingual checkpoint. Without it the language field remains null:
there is no independent language-ID output, and `DetectLanguage` refuses rather
than guessing from text. All other unsupported requests are refused through
the capability layer. Instances are not concurrent; cancellation is checked at
front-end frames, graph boundaries and decoding steps.
