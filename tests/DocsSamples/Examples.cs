// Compiled documentation examples; keep each body identical to its Markdown snippet.
using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.Media;
using OneSpeech.Qwen3;

internal static class Examples
{
    // README.md snippet 1
    internal static async Task Example1()
    {

Transcript transcript = Speech.Transcribe("meeting.m4a", language: "en");
Console.WriteLine(transcript.Text);
File.WriteAllText("meeting.json", transcript.ToJson());

await foreach (Segment segment in Speech.TranscribeStream("lecture.wav"))
    Console.WriteLine(segment.Text);

Transcript aligned = Speech.Align("short.wav", "The words that were spoken.", language: "en");
    }

    // docs/diarization.md snippet 1
    internal static async Task Example2(SpeechRecognizer recognizer, VoiceActivityDetector vad, string path)
    {
using var overlap = await OneSpeech.Diarization.NemotronDiarizer.LoadAsync();
var transcript = new OneSpeech.Pipeline.VadTranscriptionPipeline(recognizer, vad, null, null, overlap).Transcribe(path, language: null, CancellationToken.None);
    }

    // docs/live-sessions.md snippet 1
    internal static async Task Example3()
    {
static async Task ShowLive(IAsyncEnumerable<OneSpeech.AudioBuffer> pcm,
    CancellationToken cancellationToken)
{
    await foreach (var update in OneSpeech.Speech.Listen(pcm, language: "en", cancellationToken: cancellationToken))
    {
        switch (update)
        {
            case OneSpeech.PartialEvent partial:
                Console.WriteLine($"Draft {partial.Revision}: {partial.Segment.Text}");
                break;
            case OneSpeech.FinalEvent final:
                Console.WriteLine($"Final: {final.Segment.Text}");
                break;
            case OneSpeech.EndpointEvent endpoint:
                Console.WriteLine($"Endpoint at {endpoint.End}: {endpoint.Reason}");
                break;
            case OneSpeech.SpeakerTurnEvent speaker:
                Console.WriteLine(speaker.Turn.Speaker.Id);
                break;
        }
    }
}
_ = (Func<IAsyncEnumerable<AudioBuffer>, CancellationToken, Task>)ShowLive;
    }

    // docs/media.md snippet 1
    internal static async Task Example4()
    {

MediaInput.Register();
var transcript = Speech.Transcribe("meeting.mkv");
var alignment = Speech.Align("meeting.mp4", "the supplied words", language: "en");
await foreach (var segment in Speech.TranscribeStream("lecture.webm"))
    Console.WriteLine(segment.Text);
await using var encoded = File.OpenRead("meeting.mkv");
var streamedInput = await Speech.TranscribeAsync(encoded, cancellationToken: CancellationToken.None);
    }

    // docs/media.md snippet 2
    internal static void Example5(Transcript transcript)
    {
var cues = transcript.ToSubtitles();
var output = RecursiveAI.OneMedia.Captions.SubtitleFormats.Ttml.Write(cues);
Console.WriteLine(output.Report); // fidelity losses, if any
File.WriteAllText("meeting.ttml", output.Text);
    }

    // docs/models.md snippet 1
    internal static async Task Example6(CancellationToken cancellationToken)
    {
var store = new OneSpeech.Catalog.ModelStore(cacheDirectory: "model-cache");
var model = await store.PullAsync("openai/whisper-tiny", cancellationToken: cancellationToken);
// model.Directory: verified publisher files; model.Info: revision, digests and licence provenance.
    }

    // docs/models.md snippet 2
    internal static void Example7()
    {
var source = new OneSpeech.Catalog.DirectoryModelSource("publisher-files");
var store = new OneSpeech.Catalog.ModelStore(source: source);
    }

    // docs/moonshine.md snippet 1
    internal static async Task Example8(IAsyncEnumerable<AudioBuffer> pcm)
    {
await foreach (Segment segment in Speech.TranscribeStream(
    "lecture.flac", model: "moonshine-ai/moonshine-base"))
    Console.WriteLine(segment.Text);

// pcm is an application-owned IAsyncEnumerable<AudioBuffer>, mono 16000 Hz.
await foreach (LiveEvent update in Speech.Listen(
    pcm, model: "moonshine-ai/moonshine-streaming-tiny"))
    if (update is PartialEvent partial) Console.Write(partial.Segment.Text);
    }

    // docs/nemo-and-transducers.md snippet 1
    internal static void Example9()
    {
var transcript = Speech.Transcribe("recording.flac", model: "nvidia/parakeet-tdt-0.6b-v3");
    }

    // docs/qwen3.md snippet 1
    internal static void Example10()
    {
var document = Speech.Transcribe("speech.flac", model: "Qwen/Qwen3-ASR-0.6B");
    }

    // docs/qwen3.md snippet 2
    internal static void Example11(string modelDirectory, AudioBuffer audio, CancellationToken cancellationToken)
    {
using var recognizer = Qwen3Recognizer.Load(modelDirectory, new Qwen3Options
{
    Context = "Vocabulary: Recursive AI, log-mel, transcription.",
    Precision = Qwen3Precision.Binary32,
    Threads = 4,
}, cancellationToken);
Transcript result = recognizer.Transcribe(audio, "en", cancellationToken);
    }

    // docs/qwen3.md snippet 3
    internal static void Example12(string alignmentDirectory, AudioBuffer audio, string text, CancellationToken cancellationToken)
    {
using var aligner = Qwen3ForcedAligner.Load(alignmentDirectory, threads: 4, cancellationToken: cancellationToken);
Transcript aligned = aligner.Align(audio, text, "en", cancellationToken);
    }

    // docs/speaker-recognition.md snippet 1
    internal static void Example13()
    {
static void RecognizeSpeaker(OneSpeech.SpeakerEmbedder embedder,
    OneSpeech.AudioBuffer enrollment, OneSpeech.AudioBuffer query,
    CancellationToken cancellationToken)
{
    using var registry = new OneSpeech.Speakers.InMemorySpeakerRegistry(embedder, threshold: 0.3);
    registry.Enroll("alice", enrollment, cancellationToken);
    var match = registry.Identify(query, cancellationToken);
    double score = registry.Verify(enrollment, query, cancellationToken);
    Console.WriteLine(match);
    Console.WriteLine(score);
}
_ = (Action<SpeakerEmbedder, AudioBuffer, AudioBuffer, CancellationToken>)RecognizeSpeaker;
    }

    // docs/transcribing-files.md snippet 1
    internal static void Example14()
    {

Transcript transcript = Speech.Transcribe("voicemail.m4a");
Console.WriteLine(transcript.Text);
File.WriteAllText("voicemail.json", transcript.ToJson());

Transcript small = Speech.Transcribe("voicemail.wav", language: "en", model: "openai/whisper-tiny");
string language = Speech.DetectLanguage("voicemail.mp3");
    }

    // docs/transcribing-files.md snippet 2
    internal static void Example15(string cache, ModelSource myModelSource)
    {

ModelStore.Default = new ModelStore(cacheDirectory: cache, source: myModelSource);
    }

    // docs/transcript-json.md snippet 1
    internal static void Example16()
    {
var document = new OneSpeech.Transcript
{
    Text = "Hello.",
    Language = "en",
    Words = [new OneSpeech.Word { Text = "Hello.", TimingMethod = OneSpeech.TimingMethod.None }]
};
string json = document.ToJson();
var restored = OneSpeech.Transcript.FromJson(json);
if (restored.Text != document.Text || restored.Words[0].Start is not null)
    throw new InvalidOperationException("Document round-trip changed text or invented timing.");
    }

    // docs/loading.md
    internal static async Task Loading(CancellationToken cancellationToken)
    {
        using SpeechRecognizer recognizer = await OneSpeech.Whisper.WhisperRecognizer.LoadAsync(model: "openai/whisper-tiny", cancellationToken: cancellationToken);
        using VoiceActivityDetector recognitionVad = new OneSpeech.Vad.SileroVad(cancellationToken: cancellationToken);
        using SpeakerEmbedder embedder = await OneSpeech.Embedders.CampPlusEmbedder.LoadAsync(cancellationToken: cancellationToken);
        using VoiceActivityDetector speakerVad = new OneSpeech.Vad.SileroVad(OneSpeech.Vad.SileroVadOptions.Diarization, cancellationToken);
        using Diarizer diarizer = new OneSpeech.Diarization.ClusteringDiarizer(embedder, speakerVad);
        using Aligner qwen = await OneSpeech.Qwen3.Qwen3ForcedAligner.LoadAsync(cancellationToken: cancellationToken);
        using Aligner ctc = await OneSpeech.Alignment.OmniAsrAligner.LoadAsync(cancellationToken: cancellationToken);
        var pipeline = new OneSpeech.Pipeline.VadTranscriptionPipeline(recognizer, recognitionVad) { Diarizer = diarizer, TimingAligners = [qwen, ctc] };
        _ = pipeline.Transcribe("meeting.wav", "en", cancellationToken);
        await Speech.PrefetchAsync(language: "en", model: "openai/whisper-tiny", cancellationToken: cancellationToken);
    }
}
