# CTC forced-alignment trellis

`CtcTrellis.Align` accepts row-major natural-log frame probabilities and a fixed
sequence of nonblank vocabulary IDs. It consumes every frame and requires the
complete target. Leading and trailing blank are allowed. Blank separates equal
adjacent target IDs; distinct adjacent IDs may follow without blank. The recurrence
is the blank-expanded target in Graves et al., ICML 2006, §4.1, with maximum
instead of summation to select a single path. This is forced alignment, not CTC
prefix decoding or the marginal probability of a target.

An exact predecessor tie prefers staying in the state, then moving one state,
then moving two. An exact terminal tie prefers the final token to trailing blank.
Accumulation uses double precision; emissions are float32. Negative infinity is
an impossible emission, while NaN, positive infinity and positive log values are
rejected. The caller supplies normalized log probabilities; the trellis does not
normalize them. Traceback is bounded to 16 million cells, with two score rows.
Cancellation is checked before validation and once per emission/trellis/traceback
frame.

The result explicitly distinguishes failure from an empty target. Failure has
negative-infinite score and no intervals; no prefix or guessed timing is returned.
An empty target is the all-blank path, including a successful score-zero path for
empty audio. Each successful nonempty target returns one inclusive-start,
exclusive-end frame interval per target position. Repeated IDs remain distinct.
Only frames occupied by that token count; intervening blank frames are not
attributed to either neighbor. The score is the best path's log probability, not
word confidence. Model adapters must establish frame-to-audio conversion and word
boundary behavior separately; token occupancy is not calibrated word timing.

## omniASR-CTC-300M adapter

The fp32 adapter loads the pinned publisher checkpoint as inert tensor data and
converts its 7-layer convolutional front end, 24-layer pre-normalized Transformer
and linear 9,812-character CTC head to OneModel format. It accepts mono 16 kHz
waveforms. The graph subtracts utterance mean and divides by the square root of
population variance plus `1e-5`, then uses the checkpoint's convolution and layer
normalization parameters. The feature extractor has a 400-sample receptive field
and a 320-sample stride. The even-width positional convolution is trimmed by one
frame on the right. No oracle-exported graph or executable checkpoint object is
loaded.

The model vocabulary contains unique single-character pieces, an identity
normalizer, no dummy prefix and no escaped whitespace. Written words are kept
separate from the spoken target: casing and punctuation are retained in the
result, while the alignment target is NFC-normalized, lowercased and excludes punctuation
other than apostrophes. Combining marks are retained, typographic apostrophes
become ASCII apostrophes, and hyphens become spaces. English integers from zero through
999,999,999 are expanded to cardinals internally (including comma-separated
integers). This is not a general inverse-text-normalization engine: currency,
decimal, date, acronym and language-specific number readings are not resolved.
Unknown characters prevent that written word from receiving a timing.

Word occupancy starts at the first target character's occupied frame and ends at
the last character's exclusive end frame, converted with the fixed 20 ms stride.
No blank gap is spread over words, and no hand-tuned latency offset is applied.
A constrained word whose geometric mean occupied-frame probability is below
0.01 is left untimed. This rejection heuristic is not a calibrated probability
of word correctness. `Confidence` is not manufactured; existing word confidence,
speaker and other non-timing fields are retained by the transcript overload.

A call is bounded to 60 seconds; longer input is refused with a typed capability
error and must be split with its corresponding written words. File input is
bounded while decoding, before a model download, and is downmixed to mono.
Explicit calls to `Speech.Align` use this model by default, downloading and
verifying publisher files and reproducing the catalog's converted hash on first
use. The string overload splits on whitespace. The transcript overload uses
`Words` in order without splitting them again. Corresponding segment word lists
are projected when they have the same written-word partition. Written text and
word boundaries are never reconstructed from the CTC hypothesis.

`VadTranscriptionPipeline.TimingAligners` is an ordered list of borrowed
aligners. Native timestamps win. Compatible forced aligners are tried in order
(for example Qwen3 before CTC); failed words retain existing DTW as the final
fallback. Compatibility requires `Capabilities.ForcedAlignment` and a matching
declared language. The default file transcription pipeline registers CTC. This
ordering is a fixed policy, not a claim that CTC has measured better boundary
error than DTW. Timing floors against external alignment truth remain necessary.

Encoded-stream twins (`Speech.AlignAsync`) leave the caller's stream open. They
perform asynchronous bounded I/O to a temporary seekable file, capped at 512 MiB
of encoded bytes, before synchronous PCM decoding/inference. The temporary file
is removed on completion, cancellation or failure. Unsupported numeric readings
(including decimals, currencies and leading-zero strings) are left untimed, not
silently assigned the timing of a different spoken number.
