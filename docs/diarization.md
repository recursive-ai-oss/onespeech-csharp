# Diarization

Speech recognition for .NET. Speaker labels are document-local, not biometric identities. No audio or embedding is persisted by diarization helpers.

## Word assignment

`TranscriptDiarization.Merge` is shared by diarizers. Input `ActiveTurn` values wrap original-clock turns and optional relative activity in [0,1] (default 1, not a probability). Turns require a named, assigned speaker, nonnegative start and positive duration. Null times are never filled in.

Intervals are half-open. A sweep partitions the turns at every endpoint. Where speakers are concurrent, the highest-activity turn wins; equal activity breaks by ordinal speaker ID. Covered duration in these partitions is summed per speaker for each word. The speaker covering the greatest duration wins; equal durations again break by ordinal ID. Uncovered portions do not vote. A positive-duration word can therefore be assigned even if most of its interval is silence. No coverage, null endpoints, reversed endpoints, and zero-duration words yield `SpeakerLabel.Unassigned`.

`Word.Overlap` is true only when a positive-duration part of the word intersects simultaneous turns from **different** speakers. Duplicate or overlapping turns from the same speaker do not count as overlap. Touching endpoints do not overlap. This flag is independent of which speaker wins. Merging replaces prior speaker/overlap assignments; it preserves word text, timing, confidence and timing method, document text, language and model provenance. Segment words are assigned by the same rule. The input document remains unchanged.

Original turns are ordered by start and ordinal speaker ID. Each assigned root word is attached once to the turn of its chosen speaker having greatest temporal intersection, breaking ties by earliest start. Turn text is the exact concatenation of attached word text, without inserted spacing. Segment-only words are assigned in their segments but are not added to the root word list or turn text.

`TranscriptDiarization.Exclusive` provides a separate, exclusive caption timeline: the highest-activity speaker at each instant, with the same tie rule, coalescing adjacent intervals for the same speaker. It leaves gaps uncovered and produces empty turn text/words; a caller must place captions explicitly rather than infer word timing from this view.

## Separated-party stereo

`ChannelDiarizer` uses channel labels `channel-1` and `channel-2`; it does not infer identities or distinguish multiple people on one channel. Auto-detection is deliberately conservative:

* exactly two channels;
* at least three 20 ms frames dominated by the left channel and at least three dominated by the right;
* dominated means energy at least 100 times the other channel (20 dB), above mean-square energy 1e-8;
* absolute normalized waveform dot product over the whole recording at most 0.3.

The final short frame is included. These are initial heuristic values, **not** thresholds fitted or validated on a speech development set. Silence, mono, dual-mono and one-sided conversations are inconclusive, not evidence of separation. Auto-detection returns false for them. This rule cannot prove separation: uncorrelated stereo music or alternating noises may pass. Known separated calls can set `forceSeparated: true`; known mixed audio should use a voice-clustering diarizer instead. Failure is a `CapabilityNotSupportedException`, never a silent downmix fallback.

For each channel, VAD runs on bounded, independently analyzed blocks of at most 30 seconds, retaining original-clock times. VAD state resets at block boundaries; this is not continuous-state VAD equivalence. Simultaneous channel speech is retained. Channel output metadata grows with turns, but temporary PCM does not grow with duration. The caller's in-memory input remains borrowed.

## Pipeline

The five-argument `VadTranscriptionPipeline` constructor accepts a borrowed `Diarizer` after the existing chunk-duration and progress arguments. The four-argument constructor is unchanged. In-memory transcription runs recognition on the usual downmix and diarization on the **original channels**, then merges words. Segment enumeration computes the whole in-memory diarization before emitting any final segments, so assignments are never revised after emission.

File and encoded-stream inputs with a configured diarizer currently throw `FeatureUnavailableException` before opening or spooling input. A whole-file embedding-sequence path is still needed; clustering separately in each recognition chunk would invent global speaker identities. Omitting the diarizer preserves existing bounded file transcription. The hook cannot add diarizer model provenance automatically because the shared `Diarizer` contract does not expose model information.

## Voice clustering status and overlap limitation

The tier-1 voice-clustering path is not available yet. Its intended VAD + embedding + spectral/AHC + VBx design **does not detect overlapped speech**: it assigns only one speaker to each speech frame. The overlap-aware upgrades, pyannote segmentation and Nemotron 3 Diarization, are coming as explicit tier-2 options; neither is loaded by these helpers. Channel diarization is a distinct exception because separated channels supply activity independently.

VBx is not a generic Gaussian smoothing pass on arbitrary embeddings. Landini et al., *Bayesian HMM clustering of x-vector sequences (VBx)*, arXiv:2012.14952, §§2.1–2.3 and 3.1, requires a pretrained PLDA model compatible with the embedding space. Its mean, whitening/LDA transform and between-speaker variances are model parameters, not clustering thresholds. No identity-covariance substitute is claimed to reproduce that method, and no such parameters are fitted here.

See [evaluation](eval.md) for no-collar, overlap-inclusive DER/JER and RTTM input.
