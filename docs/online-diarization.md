# Live speaker labels

`Speech.Listen` adds bounded online clustering to live recognition. The low-level
`OnlineDiarizationSession` decorates a **fresh** recognition-only `LiveSession` and
borrows its session and `SpeakerEmbedder`; dispose the decorator first, then its
borrowed resources. Push mono 16 kHz blocks of at most ten seconds. Complete both
recognition and speaker output by completing the decorator, not the inner session.
Do not push or complete the inner session separately.

At each recognized utterance closure, the retained PCM is partitioned into
nonoverlapping windows of at most 1.5 seconds. A residual tail below 45 ms shares
the preceding window's duration by splitting the last two windows evenly. A whole
utterance below 45 ms is Unassigned. Each window's normalized embedding is compared
to unit centroids by cosine similarity. The best qualifying centroid wins;
exact ties prefer the oldest ID. Otherwise a new ID is allocated. The default
minimum similarity is 0.3 (AMI development selection, frozen parameters in
`eval/manifests/online-clustering.json`). The default capacity is sixteen per
embedding space; an unmatched window at capacity stays
Unassigned instead of being forced into an existing speaker. Centroids use
bounded-weight updates (weight increases to 100, then update weight is 1/101).
No speaker count is inferred from future speech.

Consecutive 1.5 s windows of one utterance that land on the same speaker are merged into one turn before anything is emitted. Turns are never merged across utterances: the earlier turn is already emitted and emitted events do not change.

`SpeakerTurnEvent`s precede the utterance's `FinalEvent`. The final also carries `SpeakerTurns`, an immutable snapshot of the same labelled
activity, so a recognizer such as Moonshine can expose speaker labels on finals
even when it supplies no word timing. This is not a claim that an untimed word
belongs to a particular window. Its timed words
receive the speaker with the greatest positive interval overlap (earliest turn on
a tie). Untimed or uncovered words are Unassigned. Word times are not invented.
Partials remain unlabelled. Turns have empty text: this is activity attribution,
not a second transcription. Turn intervals cover the recognizer's retained
utterance, including its VAD padding, and are **not** precise word boundaries.
If recognition emits no final, it supplies no interval to diarize.

Finals and their word labels are immutable. IDs are never retrospectively merged,
renumbered, or reused. If two early IDs later appear to be the same speaker, both
remain; later windows can match either centroid. This intentionally sacrifices
offline global consistency. Do not interpret these session-local IDs as enrolled
identities. No embeddings or audio are persisted; disposal erases working arrays.

`Speech.Listen` uses CAM++ English for non-Chinese requested languages and CAM++
Chinese for Chinese (`zh`, `zh-*`, `Chinese`). With automatic language detection,
it loads both models and selects by each final's language. Their separate
embedding spaces have distinct ID prefixes; a language switch can split one
person into two IDs. Explicit language avoids loading the unused model. Downloads
follow the usual first-use cache rules.

## Latency and limits

No future audio beyond the recognizer's own utterance closure is used; clustering
adds zero audio lookahead, but adds synchronous embedding CPU work at the endpoint.
A window near the beginning of a long utterance therefore waits for that entire
utterance (by default at most about 30 seconds, plus endpoint silence and caller
chunk granularity). Input has to be fed at a sufficiently small chunk size for the
latency the application needs. Speaker updates are not low-latency change-point
detection. A duration-limited endpoint commits even if speech continues.

Storage is independent of recording length: 320 seconds of PCM, a 1.5-second
scratch window, bounded centroids and at most one utterance's output. Finals must
span at most 300 seconds and still be retained; unsupported intervals are refused,
never silently clipped. Audio-time stamps are relative to the first input sample.
The decorator forwards recognition events on that same clock and never reads wall
time. Model inference and immutable emitted snapshots allocate; centroid assignment
and non-emitting decorator pushes need no additional allocation. Endpoint scratch
and word/turn construction allocations are reported separately, not claimed zero.

## Limits versus offline diarization

This is a causal floor, not equivalent to the offline spectral clustering and
resegmentation pipeline. It cannot detect overlap, split simultaneous speakers,
use future context to repair an early decision, or reliably identify a short
speaker interjection. Fixed windows may straddle changes; VAD errors and empty ASR
output also affect coverage. Use offline diarization when final global consistency
matters. A tier-2 Nemotron end-to-end live upgrade is planned, **not currently
available**; it must be selected explicitly once supported, never silently used.

The evaluation tool `OneSpeech.OnlineDiarization` streams prepared mono WAV in
100 ms chunks through `UtteranceLiveSession` plus this same decorator. It uses an
activity-only recognizer to isolate diarization from ASR lexical failures, not
oracle speech intervals. Silero runs on the input; no reference boundaries or
speaker counts enter inference. It scores all emitted activity against the
prepared RTTM using no collar and scoring overlap. Fit similarity only on AMI
development, freeze the configuration/hash, then score holdouts. Allocation and
CPU diagnostics are indicative, not benchmark claims.
