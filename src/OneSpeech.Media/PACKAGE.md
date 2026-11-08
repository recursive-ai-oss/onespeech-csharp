# Speech recognition for .NET

`RecursiveAI.OneSpeech.Media` adds container audio input, subtitles and caption muxing to OneSpeech on .NET 10. It uses OneMedia and OneMedia.Audio 1.0.2; core remains independent of them. OneSpeech is unaffiliated with other products or companies named OneSpeech.

```csharp
using OneSpeech;
using OneSpeech.Media;

MediaInput.Register(); // once at startup
Transcript transcript = Speech.Transcribe("interview.mkv");
File.WriteAllText("interview.vtt", transcript.ToVtt());
transcript.MuxCaptions("interview.mp4", "captioned.mp4");
```

The registered adapter opens containers that OneMedia supports and decodes their first audio track when OneMedia.Audio supports its codec. Video is not decoded. Existing core path and stream transcription, alignment, diarization in transcription, and path streaming use the adapter. `MediaInput.TranscribeStream(stream)` also accepts nonseekable encoded input, spooling up to 512 MiB before streaming final segments. Streams stay caller-owned.

`ToSubtitles()` returns OneMedia's `SubtitleDocument`; `ToTranscript()` imports cues. Exports: `ToSrt()`, `ToVtt()`, `ToAss()`, `ToTtml()`. Untimed words keep null span times. Segments without any timing retain unknown cue times, not guessed ones; the string export extensions refuse those cues until aligned. SRT cannot carry word times or speakers; WebVTT carries voices/start tags, ASS carries cue speakers/karaoke, TTML carries agents and span bounds. Use OneMedia's low-level writers for fidelity/loss reports.

`MuxCaptions` copies compatible tracks from MP4/MOV/3GP to a **new MP4** without re-encoding, adding WebVTT captions. It never overwrites an existing file. No metadata-copy or codec-conversion promise is made. `EncodeCaptionTrack` exposes OneMedia text-track packets for custom MP4/Matroska muxing and its loss report.

[Caption policy and examples](https://github.com/recursive-ai-oss/onespeech-csharp/blob/main/docs/media.md) · [Source](https://github.com/recursive-ai-oss/onespeech-csharp) · [Project](https://recursiveai.org/onespeech/)
