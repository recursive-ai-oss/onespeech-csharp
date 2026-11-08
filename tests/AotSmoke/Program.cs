using OneSpeech;

var phrases = new OneSpeech.Decoding.PhraseTrie([new([1], 1)]);
var biasedCtc = OneSpeech.Decoding.ContextualCtcBeamSearch.Decode([0, -.5f, -10], 3, 2, phrases);
if (!biasedCtc.Tokens.AsSpan().SequenceEqual([1]) || biasedCtc.ContextScore != 1)
    throw new InvalidOperationException("Contextual CTC NativeAOT smoke failed.");
var biasedTdt = OneSpeech.Decoding.ContextualTransducerSearch.DecodeTdt(1, 3, 2, [0, 1],
    (int frame, ReadOnlySpan<int> prefix, Span<float> logits, Span<float> durations, CancellationToken ct) =>
    { ct.ThrowIfCancellationRequested(); logits[0] = 0; logits[1] = -.5f; logits[2] = -10; durations[0] = -10; durations[1] = 0; }, phrases);
if (biasedTdt.Length != 1 || biasedTdt[0].Id != 1)
    throw new InvalidOperationException("Contextual TDT NativeAOT smoke failed.");
Console.WriteLine("Phrase trie and contextual CTC/TDT NativeAOT smoke passed.");
TransducerSmoke.Run();
NemotronSmoke.Run();
ParakeetSmoke.Run();
await NemotronAsrSmoke.Run();
CustomSmoke.Run();

var ctc = OneSpeech.Decoding.CtcTrellis.Align([-9, 0, 0, -9, -9, 0], 3, 2, [1, 1]);
if (!ctc.Succeeded || ctc.Tokens.Length != 2 || ctc.Tokens[1].StartFrame != 2)
    throw new InvalidOperationException("CTC trellis NativeAOT smoke failed.");
Console.WriteLine("CTC trellis NativeAOT smoke passed.");
OmniSmoke.Run();

var word = new Word { Text = "test", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), TimingMethod = TimingMethod.Native, Speaker = SpeakerLabel.Unassigned };
var original = new Transcript { Text = "test", Duration = TimeSpan.FromTicks(2798640000), Words = [word] };
var copy = Transcript.FromJson(original.ToJson());
JsonSmoke.Run(original);
if (copy.Text != original.Text || copy.Words[0] != word || copy.Duration != original.Duration) throw new InvalidOperationException("Transcript round-trip failed.");

if (OneSpeech.Catalog.ModelCatalog.Default.Get("openai/whisper-tiny").Revision != "169d4a4341b33bc18d8881c4b69c2e104e1cc0af" ||
    OneSpeech.Catalog.ModelCatalog.Default.Get("iic/speech_campplus_sv_en_voxceleb_16k").ConvertedFiles[0].Sha256 != "f333a00d5d66b9bf3d7e7abf35f36ff4a01e6e8a0df235b89a125e9fbe3cc15c")
    throw new InvalidOperationException("Embedded catalog pins failed.");
if (OneSpeech.Catalog.ModelCatalog.Default.GetRoleDefault("transcription").Id != "openai/whisper-large-v3-turbo")
    throw new InvalidOperationException("Catalog transcription role default failed.");
Console.WriteLine("Embedded catalog NativeAOT parsing passed.");
using var vad = new OneSpeech.Vad.SileroVad();
if (!vad.Detect(new AudioBuffer(new float[16000], 16000), CancellationToken.None).IsEmpty)
    throw new InvalidOperationException("Silero VAD reported speech on silence.");
Console.WriteLine("Transcript JSON and embedded Silero VAD NativeAOT smoke passed.");

var diarized = OneSpeech.Diarization.TranscriptDiarization.Merge(copy,
    [new OneSpeech.Diarization.ActiveTurn { Turn = new Turn { Speaker = new SpeakerLabel { Id = "speaker-1" }, Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Text = "" } }]);
if (diarized.Words[0].Speaker?.Id != "speaker-1" || OneSpeech.Diarization.ChannelDiarizer.IsSeparated(new AudioBuffer(new float[3200], 16000, 2)))
    throw new InvalidOperationException("Diarization NativeAOT smoke failed.");
Console.WriteLine("Diarization merge and channel detection NativeAOT smoke passed.");

using var smokeEmbedder = new SmokeEmbedder(vad.Model);
var clustering = new OneSpeech.Diarization.ClusteringDiarizer(smokeEmbedder, new SmokeVad(vad.Model));
if (clustering.Diarize(new AudioBuffer(new float[3200], 16000), CancellationToken.None).Length != 1)
    throw new InvalidOperationException("Clustering NativeAOT smoke failed.");
Console.WriteLine("Clustering NativeAOT smoke passed.");

SpeakerSmoke.Run();
OnlineSpeakerSmoke.Run();
Qwen3Smoke.Run();
Console.WriteLine("Speaker arithmetic and in-memory registry NativeAOT smoke passed.");

if (args is ["--camp" or "--camp-zh", var campGraph, var campAudio])
{
    var clip = OneAudio.Audio.Load(campAudio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    bool chinese = args[0] == "--camp-zh";
    var entry = OneSpeech.Catalog.ModelCatalog.Default.Get(chinese ? "iic/speech_campplus_sv_zh-cn_16k-common" : "iic/speech_campplus_sv_en_voxceleb_16k");
    var model = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, License = entry.License, Tier = entry.Tier, Attribution = entry.Attribution };
    using var camp = new OneSpeech.Embedders.CampPlusEmbedder(campGraph, model, dimension: chinese ? 192 : 512);
    float[] vector = new float[camp.Dimension];
    camp.Embed(new AudioBuffer(clip.Samples, 16000), vector, CancellationToken.None);
    Console.WriteLine("CAM++ real graph NativeAOT extraction passed.");
}

Console.WriteLine("Transcript JSON NativeAOT round-trip passed.");
var features = new OneSpeech.Audio.WhisperLogMelSpectrogram(128).Extract(new AudioBuffer(new float[400], 16000), padToWindow: false);
if (features.Length != 256 || features.Any(value => value != -1.5f)) throw new InvalidOperationException("Whisper feature smoke failed.");
Console.WriteLine("Whisper features NativeAOT smoke passed.");
string? tinyDirectory = Environment.GetEnvironmentVariable("ONESPEECH_WHISPER_TINY");
string? clipPath = Environment.GetEnvironmentVariable("ONESPEECH_SMOKE_CLIP");
if (tinyDirectory is not null && clipPath is not null)
{
    using var recognizer = OneSpeech.Whisper.WhisperRecognizer.Load(tinyDirectory);
    var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    var transcript = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
    if (transcript.Text != " Stuffed into you, his belly, cowtled him." || transcript.Words.Length == 0) throw new InvalidOperationException("Whisper tiny NativeAOT transcription failed.");
    var conditioned = recognizer.WithPhrases([new Phrase("Stephen")]);
    if (conditioned.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None).Words.IsEmpty)
        throw new InvalidOperationException("Whisper contextual wrapper NativeAOT failed.");
    Console.WriteLine("Whisper tiny NativeAOT transcription and borrowed prompt passed.");
    await LiveSmoke.Run(vad, recognizer, clipPath);
    using var beam = OneSpeech.Whisper.WhisperRecognizer.Load(tinyDirectory, new OneSpeech.Whisper.WhisperOptions { BeamSize = 1, Temperatures = [0], WordTimestamps = false });
    if (beam.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None).Text != transcript.Text) throw new InvalidOperationException("Whisper beam-width-one smoke failed.");
    Console.WriteLine("Whisper beam NativeAOT smoke passed.");
}

if (args is ["--ecapa", var ecapaGraph, var ecapaAudio])
{
    var entry = OneSpeech.Catalog.ModelCatalog.Default.Get("speechbrain/spkrec-ecapa-voxceleb");
    if (entry.ConvertedFiles[0].Sha256 != "f107ead463783cab70914fd0470cc6b42e2ec74b7a717bb42c1360879076c669")
        throw new InvalidOperationException("ECAPA graph pin failed.");
    var clip = OneAudio.Audio.Load(ecapaAudio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    var info = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, License = entry.License, Tier = entry.Tier, Attribution = entry.Attribution };
    using var model = new OneSpeech.Embedders.EcapaTdnnEmbedder(ecapaGraph, info);
    float[] vector = new float[192];
    model.Embed(new AudioBuffer(clip.Samples, 16000), vector, CancellationToken.None);
    Console.WriteLine("ECAPA real graph NativeAOT extraction passed.");
}

string? moonshineDirectory = Environment.GetEnvironmentVariable("ONESPEECH_MOONSHINE_STREAMING_TINY");
if (moonshineDirectory is not null && clipPath is not null)
{
    using var moonshine = OneSpeech.Moonshine.MoonshineRecognizer.Load(moonshineDirectory);
    try { moonshine.WithPhrases([new Phrase("Stephen")]); throw new InvalidOperationException("Moonshine silently accepted biasing."); }
    catch (CapabilityNotSupportedException) { }
    await LiveSmoke.Run(vad, moonshine, clipPath, expectPartials: true);
}

string? happyPathClip = Environment.GetEnvironmentVariable("ONESPEECH_HAPPY_PATH_CLIP");
if (happyPathClip is not null)
{
    var document = Speech.Transcribe(happyPathClip, language: "en", model: "openai/whisper-tiny");
    if (document.Words.IsEmpty || document.Turns.IsEmpty || document.Language != "en" ||
        document.Words.Any(w => w.Start is null || w.End is null || w.Confidence is null))
        throw new InvalidOperationException("Speech.Transcribe NativeAOT document is incomplete.");
    Console.WriteLine("Speech.Transcribe tiny NativeAOT happy path passed.");
}

string? pyannoteSnapshot = Environment.GetEnvironmentVariable("ONESPEECH_PYANNOTE_SNAPSHOT");
if (pyannoteSnapshot is not null)
{
    using var segmentation = OneSpeech.Diarization.PyannoteSegmentation.LoadLocal(pyannoteSnapshot);
    float[] frames = segmentation.GetFrameLogProbabilities(new AudioBuffer(new float[160000], 16000));
    if (frames.Length != 589 * 7 || frames.Any(value => !float.IsFinite(value)))
        throw new InvalidOperationException("Segmentation NativeAOT frame shape or finiteness failed.");
    using var segmentationEmbedder = new SmokeEmbedder(segmentation.Model);
    using var segmentationDiarizer = new OneSpeech.Diarization.PyannoteDiarizer(segmentation, segmentationEmbedder);
    var segmentationTurns = segmentationDiarizer.Diarize(new AudioBuffer(new float[160000], 16000));
    if (segmentationTurns.Any(turn => turn.Start < TimeSpan.Zero || turn.End > TimeSpan.FromSeconds(10)))
        throw new InvalidOperationException("Segmentation NativeAOT turn bounds failed.");
    Console.WriteLine("Local segmentation and overlap-aware diarizer NativeAOT inference passed.");
}
string? int8WhisperDirectory = Environment.GetEnvironmentVariable("ONESPEECH_WHISPER_INT8");
if (int8WhisperDirectory is not null && clipPath is not null)
{
    var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    using var model = OneSpeech.Whisper.WhisperRecognizer.Load(int8WhisperDirectory,
        new OneSpeech.Whisper.WhisperOptions { Temperatures = [0], WordTimestamps = false });
    var result = model.Transcribe(new AudioBuffer(clip.Samples, 16000), "en", CancellationToken.None);
    if (string.IsNullOrWhiteSpace(result.Text) || !result.Models[0].Id.EndsWith("-int8", StringComparison.Ordinal))
        throw new InvalidOperationException("Whisper int8 NativeAOT transcription failed.");
    Console.WriteLine("Whisper int8 NativeAOT transcription passed.");
}

internal sealed class SmokeEmbedder(ModelInfo model) : SpeakerEmbedder
{
    public override ModelInfo Model => model;
    public override int Dimension => 2;
    public override int SampleRate => 16000;
    public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
    { cancellationToken.ThrowIfCancellationRequested(); destination[0] = 1; destination[1] = 0; }
    public override void Dispose() { }
}
internal sealed class SmokeVad(ModelInfo model) : VoiceActivityDetector
{
    public override ModelInfo Model => model;
    public override System.Collections.Immutable.ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
    { cancellationToken.ThrowIfCancellationRequested(); return [new SpeechInterval { Start = TimeSpan.Zero, End = audio.Duration }]; }
}
