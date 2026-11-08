# Custom speech graphs

`OneSpeech.Custom.GraphSpeechRecognizer` is a low-level `SpeechRecognizer` for
caller-supplied float32 ONNX or OneModel files. It never downloads or converts
weights. The caller supplies lawful provenance in `ModelInfo` and an immutable
`TokenVocabulary`, including a blank ID. Built-in family loaders remain the
simpler choice for supported checkpoints.

## Encoder contract

Each `EncoderGraph` has exactly one float32 input and output, batch one. Inputs
are `[1, channels, frames]`. Intermediate outputs and transducer encoder outputs
are channel-major. The final CTC output is `[1, frames, vocabularySize]`, raw
logits, including blank. Stages can split large encoders at layer boundaries.
Declared batch, ranks and channel dimensions are checked at load; output sizes
and temporal stride are checked at execution. Violations throw
`InvalidModelShapeException`. File/format errors remain OneModel's load errors.

Front ends accept finite mono 16000 Hz PCM:

- `RawWaveform`: unchanged `[1,1,samples]`; no normalization is inferred.
- `Whisper80` / `Whisper128`: unpadded log-mel, `samples / 160` frames.
- `Nemo128`: NeMo normalized mel, with the masked trailing STFT frame cropped.

For each stage, frames become `floor((frames - receptiveField) / stride) + 1`,
or `ceil(frames / stride)` when `SamePadding` is true. The caller specifies the
final `frameDuration` in audio time. For omniASR, use raw waveform, receptive
field 400, stride 320, 9812 output channels, blank 0 and 20 ms. For the split
Parakeet encoder, use NeMo128, 1024 channels at both stages, first-stage stride 8
with same padding, then stride 1, blank 8192 and 80 ms.

Calls below 400 samples produce no words. Calls over 30 seconds use Silero
VAD-first chunks; `TranscribeStream` yields immutable final segments from those
chunks. Language hints are output labels, not acoustic conditioning or language
identification. No translation or phrase-biasing head is inferred.

## Search and vocabulary

CTC uses greedy collapse by default; `beamSize` 1–64 enables prefix beam for
vocabularies up to 4096 IDs. Beam tokens are timed with our CTC Viterbi trellis;
greedy tokens retain their occupied frame intervals. A failed beam alignment
throws rather than inventing word timings. Confidence is the mean conditional
token probability over its occupied frames, not a calibrated word posterior.

Vocabulary pieces concatenate without normalization. Optional metaspace decoding
replaces U+2581 with space and removes one leading space. The caller supplies
special-token surfaces (including an empty blank surface), and any unknown-token
surface; byte-fallback decoding is not inferred. Pieces must not encode multiple
whitespace-separated words in a single token: disjoint token ranges are required
for word timing. Optional trailing-punctuation exclusion matches native
transducer timing without changing written text. All words carry `Native`
timing provenance, null confidence/times for a zero-token range, and initially
`Unassigned` speaker. Times are clipped to the input horizon.

## Packed transducer contract

`TransducerGraphs` declares separate predictor and joint files. The predictor
maps `[1,stateSize]` to the same shape. Its zero state is run once to prime the
blank input. On each accepted token, the first `embeddingSize` state values are
replaced by that token's row in the caller-supplied embedding table, then run
again. The first `embeddingSize` output values feed the joint after the current
encoder frame. Remaining state values carry recurrent state.

The joint maps `[1,encoderChannels + embeddingSize]` to
`[1,vocabularySize + durationCount]`. Empty durations select RNN-T greedy;
strictly increasing durations beginning at zero select TDT greedy. Token logits
precede duration logits. Default packed widths (640/3200), duration table
`[0,1,2,3,4]`, and 8193 embedding rows describe the converted Parakeet graph.
Other predictor/state layouts need an adapted graph, not an implicit shim.
All calls are synchronous, cancellation-aware at graph/search steps, and not
thread-safe; dispose the recognizer to release its owned sessions.
