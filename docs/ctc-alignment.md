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
