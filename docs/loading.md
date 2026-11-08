# Loading and composing roles

Every downloadable family has the same canonical loader pair:

- `Load(ResolvedModel model, options/threads, cancellationToken)` opens already-pulled local files without network access. Obtain the directory/provenance pair from `ModelStore.PullAsync`, not by guessing cache filenames.
- `LoadAsync(ModelStore? store = null, string? model = null, options/threads, CancellationToken cancellationToken = default)` pulls, verifies and locally converts missing files before loading. Defaults are the family's declared checkpoint, not a tier-2 default for `Speech`. Calling a named tier-2 family loader is explicit selection of its documented publisher checkpoint.

Whisper, Moonshine, Qwen3-ASR and Nemotron ASR use typed options; other loaders accept `threads`, default 1. All family options call this knob `Threads`, default 1; families permitting 0 document it as runtime-auto. String-directory overloads and graph-path constructors remain for manually converted/custom graphs and require the documented family graph format. They are not needed for catalog loads. Embedded Silero uses its constructor and never downloads.

## Rebuild the file pipeline

```csharp
using SpeechRecognizer recognizer = await WhisperRecognizer.LoadAsync(
    model: "openai/whisper-tiny", cancellationToken: cancellationToken);
using VoiceActivityDetector recognitionVad = new SileroVad(cancellationToken: cancellationToken);
using SpeakerEmbedder embedder = await CampPlusEmbedder.LoadAsync(cancellationToken: cancellationToken);
using VoiceActivityDetector speakerVad = new SileroVad(SileroVadOptions.Diarization, cancellationToken);
using Diarizer diarizer = new ClusteringDiarizer(embedder, speakerVad);
using Aligner qwen = await Qwen3ForcedAligner.LoadAsync(cancellationToken: cancellationToken);
using Aligner ctc = await OmniAsrAligner.LoadAsync(cancellationToken: cancellationToken);
var pipeline = new VadTranscriptionPipeline(recognizer, recognitionVad)
{
    Diarizer = diarizer,
    TimingAligners = [qwen, ctc]
};
Transcript result = pipeline.Transcribe("meeting.wav", "en", cancellationToken);
```

Namespaces: `OneSpeech`, `.Whisper`, `.Vad`, `.Embedders`, `.Diarization`, `.Qwen3`, `.Alignment`, `.Pipeline`. `CampPlusEmbedder.LoadAsync(model: "iic/speech_campplus_sv_zh-cn_16k-common")` derives the Chinese dimension itself. The example uses explicit English timing/embedding; automatic per-recording language selection and deferred model acquisition are provided by `Speech.Transcribe`. It eagerly loads both timing candidates, unlike the lazy happy path. It uses default clustering and resegmentation with the diarization-specific VAD profile; no overlap detection is implied.

Roles are disposable owners of **their own** graph resources. Pipelines, clustering diarizers, live sessions and recognizer decorators borrow their collaborators; disposing a wrapper/session never disposes a borrowed recognizer, detector, aligner or embedder. Dispose sessions before their models and models after all processing has stopped. `Dispose` is idempotent. Custom subclasses override `Dispose(bool)` and call the base. Progress uses `IProgress<TimeSpan>` with processed audio time, never wall time.

`SpeechRecognizer.CreateLiveSession(VoiceActivityDetector detector, language)` accepts any detector. Silero sessions own independent recurrent state; custom detectors receive one zero-padded 32 ms mono frame at a time. They should be stateless or dedicated to that session. A custom VAD returns intervals, not calibrated probabilities; any positive interval counts as active for endpointing.
