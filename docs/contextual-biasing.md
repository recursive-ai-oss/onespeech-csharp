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

## Recognizer requests

`SpeechRecognizer.WithPhrases(IEnumerable<Phrase>, CancellationToken)` copies and
validates ordinary text, then returns a borrowed recognizer. No model is loaded
again. The owner must outlive wrappers/sessions; existing single-instance
concurrency restrictions are unchanged. Wrappers are not disposable owners.
Reapplying the method replaces the phrase list, including when applied to a
wrapper. Empty lists return the original recognizer. Unbiased recognizers and
other wrappers remain unchanged. `VadTranscriptionPipeline` accepts a wrapper
without any new options. File and utterance-live chunking retain its context.
Language detection is delegated to the original recognizer without bias.

Common limits are 4096 phrases, 1024 UTF-16 units per phrase, and 262144 total
units. Text must be valid Unicode and non-whitespace; outer whitespace is removed.
Weights must be finite and strictly positive, at most 20. Model-specific limits
can be smaller. No phrases or embeddings are persisted.

- **Whisper:** comma-separated phrases encoded as ordinary text with a leading
  space; special tokens refused. The entire fixed prompt must fit 223 tokens,
  otherwise configuration throws. Each window uses `startofprev`, fixed phrase
  tokens, then the newest generated-context tokens that fit the remaining
  223-token budget, followed by task/language controls. Fixed phrases survive
  fallback context resets. Only generated tokens enter the transcript. This
  uses the prompt-conditioning mechanism described in Radford et al. (2022)
  §2.2; repeating the fixed phrases at each window is this library's policy.
  This is an opt-in hint, not constrained decoding: it can substantially worsen
  WER or repeatedly hallucinate absent prompted terms. Improved term recall alone
  is not evidence that a prompt is safe.
- **Qwen3-ASR:** append comma-separated phrases to the configured `Context`,
  separated by a newline, and use the existing system-context path. Ordinary
  tokenizer encoding validates the entire combined 4096-token budget and
  refuses special tokens. No second biasing implementation is introduced.
- **Prompt weights:** explicit numeric weights are refused with
  `CapabilityNotSupportedException` for Whisper and Qwen. Repeating words or
  adding weight text would not implement a numeric score bonus.
- **Parakeet TDT:** use the publisher's pinned BPE ranks, precompiled character
  normalization and Metaspace word boundaries. Text containing control-token
  syntax or characters that encode as unknown is refused, rather than biasing
  the shared unknown ID. Ordinary token weights feed the phrase trie.
- **Custom graphs:** `TokenVocabulary.WithPhraseEncoder` attaches the caller's
  exact encoding callback to a copy of the vocabulary. No segmentation or
  normalization is guessed from piece lengths. Nonblank ordinary IDs are the
  caller's contract; blank/out-of-range IDs and trie limits are validated.
  Contextual CTC requires at most 4096 columns and uses the configured beam or
  beam 8 when unbiased decoding is greedy. RNN-T/TDT remain greedy.
- **Unsupported families:** Moonshine retains `Biasing=false`; nonempty phrases
  throw `CapabilityNotSupportedException` before recognition, never silently
  ignored. Empty requests are neutral. Invalid/canceled requests still fail.

## Targeted evaluation

`tools/OneSpeech.BiasingEval` prepares a fixed small Earnings-22 comparison. All
audio, references, hypotheses and phrase lists stay in an external state directory,
never committed (Earnings-22 CC-BY-SA-4.0). The original fetch script
`tools/scripts/fetch-earnings22.sh` pins the publisher revision and input hashes.
The diagnostic additionally verifies its five input SHA-256 pins in
`tools/scripts/pins/biasing-earnings22.sha256` before decoding.

Selection is outcome-independent: the two shortest complete recordings by the
pinned metadata's numeric duration, ordinal file-ID tie-break. These are
4474955 (875 seconds) and 4483046 (1187 seconds), about 34 minutes total. Full
human references avoid guessed crop boundaries or alignment-dependent labels.
Phrase candidates have at least five ASCII letters, 1–3 case-insensitive
occurrences in that file's full reference, and at least one capitalized occurrence
away from a sentence start. Select at most 12 in ordinal case-insensitive order.
The scorer requires each selected term to normalize to one word and fails
preparation otherwise. Controls swap the other file's selected list, removing
terms present anywhere in the normalized control reference. This reference-derived
list is a controlled diagnostic, not an estimate of how users choose phrases.

From the repository root, run `prepare CORPUS STATE`, then each
`run FAMILY MODEL STATE CASE MODE` for families `whisper`, `qwen`, `parakeet`,
cases `0`, `1`, and modes `baseline`, `phrases`, `control`. `score STATE` writes
`scores.jsonl`. Model directories are existing publisher-verified converted caches,
never modified. Shard by (family, case, mode); atomic transcripts allow bounded
jobs to resume without rerunning completed cases.

Every mode uses the same public `VadTranscriptionPipeline`, default Silero VAD,
30-second maximum chunks and English hint. Whisper-small uses fp32, temperature
zero and no DTW; Qwen3-ASR-0.6B uses fp32 greedy; Parakeet uses fp32 greedy TDT.
Each uses four graph threads. All fusion weights retain the unmodified default 1;
prompts have no explicit weights. No model parameters or score weights are fitted
on these files.

Both sides use `en/1` normalization and the same word edit alignment. Phrase
recall counts correctly aligned reference occurrences, not bag-of-word presence.
Control false alarms count normalized occurrences of absent control terms in
the hypothesis. Report WER and occurrence counts/recall for all modes per file,
including control WER and its difference from baseline. These two files are
conditional diagnostic floors, not corpus confidence intervals or a general
quality claim; do not aggregate away a per-file regression.

`bootstrap.jsonl` also records paired file-level 95% percentile intervals (1000
resamples, seed 1729) for WER and phrase miss rate (`1 - recall`), comparing each
condition with baseline. With only two files these intervals are very coarse
and conditional on this selection; they do not replace the per-file rows.
