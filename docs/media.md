# Container audio and captions

`RecursiveAI.OneSpeech.Media` references core, OneMedia and OneMedia.Audio. Call `OneSpeech.Media.MediaInput.Register()` once during startup, before inference. Registration sets core's explicit `AudioFileSource.AdditionalDecoder` hook; it does not scan assemblies, download models or change the document schema. OneAudio and the core G.711 reader continue handling their formats first. On an unsupported format the stream is rewound and OneMedia opens it. Containers without a decodable audio track fail rather than returning silence. OneMedia exceptions retain their typed details.

## Audio input

After registration:

```csharp
using OneSpeech;
using OneSpeech.Media;

MediaInput.Register();
var transcript = Speech.Transcribe("meeting.mkv");
var alignment = Speech.Align("meeting.mp4", "the supplied words", language: "en");
await foreach (var segment in Speech.TranscribeStream("lecture.webm"))
    Console.WriteLine(segment.Text);
await using var encoded = File.OpenRead("meeting.mkv");
var streamedInput = await Speech.TranscribeAsync(encoded, cancellationToken: CancellationToken.None);
```

The normal transcription pass includes diarization; core's low-level audio source exposes the same demuxed/resampled PCM to custom pipelines. The adapter uses OneMedia's default audio track and codecs, not video decoding. Supported containers include MP4/MOV, Matroska/WebM, AVI, transport streams, ASF and FLV, subject to OneMedia.Audio codec support. No claim is made that every codec inside those containers is decodable. Core downmixing, resampling, duration limits, timing and cancellation policies are unchanged. Disposing the adapter decoder releases its media file, not a caller-owned stream.

For nonseekable encoded input, `MediaInput.TranscribeStream(stream, language: "en", cancellationToken: token)` asynchronously copies at most 512 MiB to a temporary disk file, then yields core streaming final segments. The copy uses a 64 KiB buffer. It does not produce live partials while copying, nor run whole-file diarization. Cancellation, errors and early enumeration disposal remove the temporary file; the caller's stream is never closed. For larger files use a local path.

## Caption policy

`transcript.ToSubtitles()` returns OneMedia's cue document. Segments define the caption groups, or root words if no segments exist. With fully timed words a group splits at speaker changes, 84 UTF-16 characters or six seconds; lines wrap at 42 characters without splitting a word. A single long word or indivisible long interval can exceed those targets. A segment with any untimed word stays one wrapped cue: there is no evidence for extra split boundaries. Text without words stays a segment cue.

Every word becomes a UTF-16-range `SubtitleSpan`, carrying its measured absolute `WordStart`/`WordEnd`, language and speaker ID as `Voice`. Null bounds remain null. Cue bounds use the segment when untimed words are present, otherwise measured word bounds. Completely untimed cues remain `MediaTime.Unknown`; track encoders drop them and report the loss. They are not silently given an arbitrary duration. The four string export extensions refuse unknown cue bounds with CapabilityNotSupportedException; align those segments first. Untimed words inside an otherwise timed segment can still be exported.

When turns exist, core's majority-overlap assignment determines word speakers (equal activity, ordinal speaker-ID tie break); uncovered words remain unassigned. Cue `Speaker` is the first word's assigned speaker. Display cues are sorted by start; overlaps end at the next cue's start and fully covered display cues are omitted. This is the exclusive caption view, not a mutation of the original Transcript. Absolute word measurements are retained even when a display interval is clipped. Untimed cues are kept after timed cues in the in-memory document.

`document.ToTranscript()` imports cue text and bounds as segments, non-raw spans as words and cue speakers as turns. It never infers word times from cue bounds. Subtitle formats can merge spans, omit untimed span boundaries, round times and drop provenance. Imported timed words use `TimingMethod.Native` and null confidence; untimed spans use `None`. This is a subtitle interchange, not lossless Transcript serialization: use `ToJson()` for that.

```csharp
var cues = transcript.ToSubtitles();
var output = RecursiveAI.OneMedia.Captions.SubtitleFormats.Ttml.Write(cues);
Console.WriteLine(output.Report); // fidelity losses, if any
File.WriteAllText("meeting.ttml", output.Text);
```

`ToSrt`, `ToVtt`, `ToAss` and `ToTtml` are convenience extensions over the same writers. SRT carries neither speakers nor word times. WebVTT carries voices and word starts (ends are reconstructed by the reader); ASS uses cue-level speaker and centisecond karaoke; TTML carries agents and timed spans. Consult the OneMedia conversion report when fidelity matters.

## Caption muxing

`transcript.MuxCaptions(input, output)` copies compatible audio/video/subtitle tracks from MP4/MOV/3GP to a new moov-last MP4 and adds a WebVTT track. It copies packets in each track's decode order; it does not re-encode, repackage unsupported codecs, copy container metadata or promise optimal interleaving. Non-ISO input is refused to avoid copying incompatible packet framing; use OneMedia remuxing first. OneMedia's typed mux refusals are preserved. Output must not exist. Failure removes the incomplete file created by the call, never an existing destination.

For a caller-managed muxer, `EncodeCaptionTrack(TextTrackFormat.WebVtt)` returns the Matroska track declaration, packets and losses. Declare all tracks first, then write each caption packet using the index returned by `AddTrack`, then `Finish`. Use `WebVttIso` (the default) for MP4. OneMedia owns the format mapping; this package contains no duplicate container reader, subtitle parser or muxer.
