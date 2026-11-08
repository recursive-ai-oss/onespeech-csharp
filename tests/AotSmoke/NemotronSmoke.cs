using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.Diarization;

internal static class NemotronSmoke
{
    internal static void Run()
    {
        var entry = ModelCatalog.Default.Get("nvidia/Nemotron-3-Diarization");
        if (entry.Tier != ModelTier.Tier2 || entry.IsDefault || entry.Capabilities.MaxSpeakers != 8 || entry.ConvertedFiles.Length != 0)
            throw new InvalidOperationException("Nemotron tier-2 catalog rules failed.");
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_NEMOTRON_DIARIZATION");
        if (directory is null) return;
        var info = new ModelInfo { Id = entry.Id, Source = entry.Source, Revision = entry.Revision, Tier = entry.Tier, License = entry.License, Attribution = entry.Attribution };
        string? cache = Environment.GetEnvironmentVariable("ONESPEECH_NEMOTRON_FRESH_CACHE");
        using var model = cache is null ? NemotronDiarizer.Load(new ResolvedModel(directory, info), threads: 4) :
            NemotronDiarizer.LoadAsync(new ModelStore(cache, new FlatSource(directory)), threads: 4).AsTask().GetAwaiter().GetResult();
        var turns = model.Diarize(new AudioBuffer(new float[1600], 16000), CancellationToken.None);
        if (turns.Any(t => t.Start < TimeSpan.Zero || t.End > TimeSpan.FromMilliseconds(100) || t.End <= t.Start))
            throw new InvalidOperationException("Nemotron diarization clock failed.");
        if (!model.Models[0].Attribution.Contains("OpenMDW License Agreement", StringComparison.Ordinal))
            throw new InvalidOperationException("Nemotron licence provenance missing.");
        var pcm = new AudioBuffer(new float[1600], 16000);
        var transcript = new OneSpeech.Pipeline.VadTranscriptionPipeline(new ProbeRecognizer(model.Model), new SmokeVad(model.Model), null, null, model).Transcribe(pcm, null, CancellationToken.None);
        if (transcript.Text != "probe" || transcript.Words.Length != 1 || transcript.Words[0].Speaker is null || !transcript.Models.Any(m => m.Id == entry.Id && m.Tier == ModelTier.Tier2))
            throw new InvalidOperationException("Nemotron pipeline assignment or provenance failed.");
        Console.WriteLine("Nemotron tier-2 graph, cache, pipeline and output NativeAOT smoke passed.");
    }
    private sealed class ProbeRecognizer(ModelInfo model) : SpeechRecognizer
    {
        public override ModelInfo Model => model;
        public override Capabilities Capabilities { get; } = new();
        public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken) => "en";
        public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
        {
            var word = new Word { Text = "probe", Start = TimeSpan.Zero, End = audio.Duration, TimingMethod = TimingMethod.Native };
            return new Transcript { Text = "probe", Words = [word], Segments = [new Segment { Text = "probe", Start = TimeSpan.Zero, End = audio.Duration, Words = [word] }], Models = [Model], Duration = audio.Duration };
        }
        public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken) => Transcribe(audio, language, cancellationToken).Segments;
    }
    private sealed class FlatSource(string root) : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<Stream>(new FileStream(Path.Combine(root, file.Name), FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous));
        }
    }
}
