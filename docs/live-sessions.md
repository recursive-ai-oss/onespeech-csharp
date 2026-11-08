# Live recognition and file streaming

A `LiveSession` accepts borrowed mono 16 kHz `AudioBuffer` blocks through
synchronous `Push`, or awaits those blocks through `Listen`. Resampling and
capture belong to the caller. `Complete` flushes real input; it does not append
silence. Normal async end-of-input calls `Complete`. Cancellation and early
iterator disposal do not flush. Sessions are single-use and not thread-safe.
Dispose them before disposing their borrowed models.

## Events and clock

All times are audio times relative to the first sample. The clock is computed
from an integer count of accepted samples, including silence. Event `AudioTime`
is the amount of input accepted when the update became available, not a word
boundary or a wall-clock measurement. A large push can return several events
with the same `AudioTime`. Applications needing finer availability measurements
should push smaller blocks. Every segment/word/turn boundary must lie within
accepted audio. Unknown word boundaries and confidence remain null.

* `PartialEvent` replaces the previous partial for its `UtteranceId`. Revision
  numbers start at one and strictly increase within that utterance.
* `FinalEvent` commits one immutable `Segment`. It cannot be revised. The shape
  of the segment is the same as file-streaming output.
* `EndpointEvent` closes an utterance after its final, if any, increments the
  utterance ID and resets the revision sequence. Its `End` is the retained audio
  boundary, not an inferred word boundary. Reasons are silence, duration limit
  and end-of-input.
* `SpeakerTurnEvent` describes activity separately and does not mutate finals.
  The utterance session does not perform speaker assignment.

The `LiveSession` base class assigns utterance IDs, revisions and availability
clock values. It rejects out-of-clock boundaries, duplicate finals and partials
after a final. A processing failure faults the session: create a new one rather
than retrying an ambiguously consumed block. Input validation and cancellation
before processing do not advance the clock.

## Offline-family utterance mode

`SpeechRecognizer.CreateLiveSession` defaults to `UtteranceLiveSession`, which
borrows a Silero model and creates independent recurrent state. It emits finals
only, not buffered-recompute partials. `Speech.Listen(source)` resolves the
selected catalog model and owns the models/session during enumeration. Model
resolution and source I/O are awaited; in-memory recognition is synchronous.

VAD uses 512-sample frames, carries incomplete frames across pushes and pads
only the last incomplete VAD frame at completion. Padded samples never advance
the clock or enter recognition. Default activation/deactivation thresholds are
0.5/0.35, minimum speech 250 ms, endpoint silence 100 ms, padding 30 ms. Speech
must exceed the minimum duration. Silence has hysteresis. Adjacent utterance
padding never overlaps already finalized audio.

PCM storage is bounded by the maximum utterance length (default 30 s; configurable
1–300 s), padding and two VAD frames. The duration limit closes at the next VAD
frame, at most 32 ms beyond the configured limit. Continuous speech continues
in the next utterance without dropping samples. Explicit completion retains a
trailing silence shorter than the endpoint threshold. Final segment bounds
name the retained utterance; only recognizer-provided word timings are shifted
onto the session clock. No word timing is fabricated from VAD bounds.

Steady-state session buffering reuses arrays, but inference, recognition,
immutable events and output arrays can allocate. Zero-allocation inference is
not claimed.

## File streaming

`Speech.TranscribeStream(path)` lazily loads models and yields final segments
through the same session used for live input, decoding 4096-frame blocks and
averaging channels. For the same decoded mono PCM, file and live modes emit
identical final segment content and boundaries, regardless of push boundaries.
Availability times can differ because different block sizes deliver the
endpoint at different accepted audio times. Early enumeration disposal closes
the decoder and models. Unlike whole-file `Transcribe`, it does not run a
whole-file diarization pass or invent speaker labels. The lower-level `VadTranscriptionPipeline` remains available with its own
chunking policy; whole-file `Transcribe` continues to use that pipeline.
Neither streaming entry point retains the whole recording's PCM.
