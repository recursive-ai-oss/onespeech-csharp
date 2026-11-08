# Contextual token scoring

`OneSpeech.Decoding.PhraseTrie` is an immutable, inference-only context graph.
Construct it from `TokenPhrase` values tokenized by the model's tokenizer. Include
word-boundary markers when the tokenizer uses them; this primitive matches IDs,
not words, case variants or Unicode strings. Blank and control tokens are never
phrases. The decoder validates vocabulary bounds and excludes blank.

A phrase's weight is a natural-log bonus per token, default 1, strictly positive
and at most 20. Completion receives `token count × weight`. Duplicate phrases
use the largest weight. Multiple phrases finishing at the same token receive
the largest completion reward, not their sum. Overlapping completions at
different positions each receive credit. Lists are bounded to 4096 phrases,
256 tokens per phrase and 65536 total tokens.

For search pruning, unfinished prefixes borrow credit proportional to their
depth and weight. Failure links retain suffix matches. `Advance` returns the
completion reward plus the change in prefix credit; `Finish` repays all remaining
prefix credit. Consequently `Score` of a complete sequence counts only completed
phrases. A search must keep one context state per hypothesis and advance it
only when accepting a new nonblank token. Do not advance on CTC repeated frames
or blank frames. Context states belong to the trie that created them.

`ContextualCtcBeamSearch` sums acoustic blank/nonblank path probabilities just as
ordinary CTC prefix search does. It ranks prefixes by acoustic log probability
plus context score before pruning, then removes unfinished credit before final
ranking. It reports acoustic probability and completion reward separately; the
fused score is not a probability. With an empty trie it is equivalent to
`CtcPrefixBeamSearch` at the same beam size. Pruning can discard the global winner.

`ContextualTransducerSearch` adds context deltas before greedy token selection
without renormalizing the acoustic distribution. Blank retains context, the
predictor receives only accepted nonblanks, and TDT duration logits are unchanged.
Confidence is always the original conditional acoustic probability. The greedy
search cannot recover an alternative after pruning: unfinished-prefix credit
can influence the final token. It shares symbol bounds and native frame intervals
with the non-contextual RNN-T/TDT decoders. Empty context is neutral.

This follows the inference-only shallow-fusion principle in Le et al.,
*Contextualized Streaming End-to-End Speech Recognition with Trie-Based Deep
Biasing and Shallow Fusion* (2021), §3.1. It does not implement that paper's learned
deep-biasing modules. Context bonuses can introduce false alarms; measure both
phrase recall and WER, including controls whose phrases are absent from the audio.
These primitives alone do not enable phrase requests on a recognizer family.
