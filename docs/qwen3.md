# Qwen3-ASR and forced alignment

Qwen3-ASR-0.6B and Qwen3-ASR-1.7B run on OneModel, including their AuT audio
encoder/projector and Qwen language decoder with a past/present KV cache.
The published ungated checkpoints are Apache-2.0. Model revisions, publisher
file SHA-256s and independently reproduced runtime-file SHA-256s are pinned
in the catalog. Weights are not embedded in the core package.

```csharp
var document = Speech.Transcribe("speech.flac", model: "Qwen/Qwen3-ASR-0.6B");
```

This file path uses VAD-first chunks, language identification, clustering
speaker attribution and Qwen3-ForcedAligner for its eleven languages. The
Qwen entries do not change the default ASR model. Missing tier-1 publisher
files download on first use; verified tensors are converted locally to
OneModel's format. No Python, native neural runtime or executable model-repo
code is needed by the library.

## Low-level recognition

```csharp
using var recognizer = Qwen3Recognizer.Load(modelDirectory, new Qwen3Options
{
    Context = "Vocabulary: Recursive AI, log-mel, transcription.",
    Precision = Qwen3Precision.Binary32,
    Threads = 4,
}, cancellationToken);
Transcript result = recognizer.Transcribe(audio, "en", cancellationToken);
```

Input is 16 kHz mono float PCM. File decoding/downmixing/resampling belongs to
OneAudio and `AudioFileSource`. More than 30 seconds uses VAD-first chunks;
each chunk has fresh decoder state. Empty audio returns an empty document;
unpadded nonempty audio must exceed 200 samples. Nonfinite samples and inputs
whose power exceeds float32 range are rejected.

Recognition is greedy. `MaxNewTokens` defaults to 512 per chunk (1–4096);
reaching the limit throws rather than returning silently truncated text.
`Context` is the system-context prompt for biasing, at most 4096 tokenizer
IDs. Special tokens in user context are refused. A language tag such as
`en`, `zh`, `fr` or `yue` forces that language; null requests the model's own
`language ...<asr_text>` output. Unsupported tags are refused. Translation
and true live partials are not declared by this family.

Low-level recognition does not download a timing model. Its word timings
remain null unless a timing stage supplies them. File transcription uses the
shared timing dispatcher, selecting Qwen3-ForcedAligner before CTC/DTW for its
eleven supported languages. Models are loaded only when needed, not attached
eagerly to the recognizer.

## Forced alignment

```csharp
using var aligner = Qwen3ForcedAligner.Load(alignmentDirectory, threads: 4, cancellationToken: cancellationToken);
Transcript aligned = aligner.Align(audio, text, "en", cancellationToken);
```

Supported tags are `zh`, `en`, `yue`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru`
and `es`. Language is required. Audio must exceed 200 samples and not exceed
300 seconds of 16 kHz mono PCM; text is limited to 1 MiB UTF-8 and 4096 units.
The released config defines 5000 timestamp classes with an 80-ms class step.
This config, rather than the report's illustrative class count, defines the
loaded head. The model predicts start/end slots in one forward pass, not by
autoregressive generation.

Whitespace delimits written units outside Chinese/Cantonese/Japanese, where
Unicode characters delimit units. Letters, numbers and ASCII apostrophes
form the spoken input; other punctuation is removed without inserting spaces.
Written whitespace-delimited forms are retained in words, and the document's
text is preserved. Combining marks removed by this cleaning do not reappear
through the tokenizer's later NFC normalization. These English unit rules
are checked against public toolkit output, including apostrophes and hyphens.
Character units do not claim the reference toolkit's language-specific
morphological segmentation.

Valid predictions carry `TimingMethod.ForcedAligner`. Predictions with reversed
bounds or an end beyond the audio duration retain null times. Confidence is
the mean of the two slot-class probabilities, not a calibrated probability
that the written word was spoken. The transcript overload instead preserves
existing word confidence and other non-timing fields, projecting character slots
back to the caller's written-word boundaries. No boundaries are interpolated or invented.
Independent slot predictions are not repaired for monotonicity across words;
valid intervals may overlap. These direct model predictions do not promise
identity with the toolkit's postprocessed boundaries.

## Front end and graph numerics

The pinned preprocessors request 128-bin Whisper features: 16 kHz mono,
400-sample periodic Hann windows, 160-sample hops, centered reflection, Slaney
mel filters, power floor 1e-10, log10 peak-minus-eight clipping, then
`(x + 4) / 4`. Mel-major single-input output has `sample_count / 160` frames,
dropping the final STFT frame. Maximum absolute feature tolerance is 3e-5
against the published float32 feature extractor. The public Whisper extractor
retains its 30-second bound; the shared internal alignment path allows the
aligner's bounded 300-second input.

Convolution chunks have 100 mel frames, padded independently, and yield up to
13 audio tokens each. Sinusoidal positions reset at each convolution chunk.
Learned activation arithmetic, including positional addition and audio-token
embedding substitution, is performed in OneModel graphs. The released CPU
toolkit emits attention-window metadata, but changing frames beyond an
eight-second boundary changes earlier encoder outputs. The CPU graphs
therefore use full-input encoder attention within each bounded input. This
is not a claim about a FlashAttention backend's window policy.

First pull converts and verifies only the Binary32 runtime graphs. BFloat16 files
remain catalogued but are not required or included in the resolved FP32 file list.
Loading `Qwen3Recognizer` with `Precision = Qwen3Precision.BFloat16` verifies any
existing BF16 graphs; if any are missing, it verifies the pinned publisher files,
converts BF16 into a staging directory, checks every output size and SHA-256, then
installs them in the same cache directory. This load never downloads. Corruption
throws `ModelIntegrityException`; remove the named file and retry. Concurrent
verification or conversion of that directory throws `IOException`; retry after it completes.
FP32 remains the default and never converts or validates unrelated BF16 files.

`Binary32` expands the original bfloat16 tensors exactly and retains float32
activations. `BFloat16` stores published tensors as bfloat16 and rounds neural
operator boundaries to nearest-even bfloat16; OneModel accumulates in float32.
Attention score/softmax intermediates remain float32, with the attended output
rounded to bfloat16. This does not promise bit-identical fused-kernel output
across runtimes. Decoder files contain four layers each; embedding and output
heads are separate files so every artifact fits OneModel's file-size limit.

## Tokenizer

`Qwen3Tokenizer` loads `vocab.json`, `merges.txt` and `tokenizer_config.json`
without executing model-repository code. Each file is limited to 16 MiB;
vocabulary IDs and merge counts are below 200,000. Input text is limited to
1 MiB UTF-8; invalid UTF-16 is rejected. It adds no implicit task tokens.

Text undergoes Unicode NFC normalization before byte-level BPE. Splitting is
Unicode-scalar-aware, including astral letters, individual numeric scalars,
case-insensitive contractions and significant whitespace/newline grouping.
Merge rank ties go left-to-right. Added tokens are atomic. Encoding tokens
marked special by model data requires permission. Omission during decoding
matches the released toolkit's Transformers 4.57.6: the named special-token
list plus BOS/EOS/padding/unknown tokens, rather than every added-token flag.
Thus `<tts_text_eod>`, blank tokens, `<asr_text>` and the aligner's `<timestamp>`
remain when special tokens are omitted. Transformers 5.22.2 differs on blank
and `<tts_text_eod>` omission; it is not the decoding reference for this API.
Unknown IDs are rejected; incomplete UTF-8 decoding uses replacement characters.

## Reproduction

`tools/oracles/qwen3/` contains the locked tokenizer/feature observations.
`tools/oracles/qwen3-asr/` uses released `qwen-asr` 0.0.6 and its required
Transformers 4.57.6 through public APIs, with CPU-only Torch. Public tensor
hooks export parity observations, never implementation source. Fixed slices
are the first 200 sorted IDs of each LibriSpeech test split. Generated audio,
features, hypotheses and transcript-derived observations stay outside git;
the committed tokenizer fixture contains only synthetic strings.

`tools/OneSpeech.Convert` provides `qwen-fetch`, `qwen-convert`,
`qwen-convert-bf16`, `qwen-reproduce`, probe and fixed-slice commands. Fetching,
weight expansion and graph construction are C#. The runtime uses the same
converter and checks converted sizes/digests before installing any graphs.

## Offline Qwen memory lifetime

Qwen file transcription recognizes all bounded VAD chunks first, recording only
chunk boundaries and written-word ranges with the transcript. It releases the
recognizer graphs before loading timing graphs, then replays decoded audio
through a fresh VAD with the same settings. Replay boundaries must match exactly;
a mismatch is an error, never a shifted word assignment. Timing is projected
back onto the original words and segments on the same audio clock. This adds a
bounded decoding/VAD pass, not a second recognition pass. Encoded-stream input
replays its existing temporary spool. No full-file PCM or KV history is retained.
Live and file-segment streaming retain their one-pass session policy; they
cannot defer finals until the end of the file. Other whole-file recognizers
without native timing use the same lifetime separation. Decoder shape caches retain
one plan: an older KV length is not reused by autoregressive generation.
