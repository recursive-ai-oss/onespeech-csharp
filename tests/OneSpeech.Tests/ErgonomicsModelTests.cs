using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class ErgonomicsModelTests
{
    [Fact]
    public async Task CatalogLoadersAndTranscriptLanguageAliasesAgree()
    {
        string? cache = Environment.GetEnvironmentVariable("ONESPEECH_API_MODEL_CACHE");
        Assert.SkipWhen(cache is null, "Set ONESPEECH_API_MODEL_CACHE to a prepopulated catalog cache for the model composition check.");
        var token = TestContext.Current.CancellationToken;
        // A missing companion must refuse rather than reaching the network during this check.
        var store = new ModelStore(cache, new DirectoryModelSource(Path.Combine(Path.GetTempPath(), "absent-model-source")));
        using SpeechRecognizer whisper = await Whisper.WhisperRecognizer.LoadAsync(store, "openai/whisper-tiny", cancellationToken: token);
        using Aligner qwen = await Qwen3.Qwen3ForcedAligner.LoadAsync(store, threads: 4, cancellationToken: token);
        using Aligner ctc = await Alignment.OmniAsrAligner.LoadAsync(store, threads: 4, cancellationToken: token);
        using SpeakerEmbedder embedder = await Embedders.CampPlusEmbedder.LoadAsync(store, threads: 4, cancellationToken: token);
        Assert.Equal(512, embedder.Dimension);
        Assert.Equal("facebook/omniASR-CTC-300M", ctc.Model.Id);
        using VoiceActivityDetector vad = new Vad.SileroVad(cancellationToken: token);
        using Diarizer diarizer = new Diarization.ClusteringDiarizer(embedder, vad);
        var pipeline = new Pipeline.VadTranscriptionPipeline(whisper, vad) { Diarizer = diarizer, TimingAligners = [qwen, ctc] };
        var silence = new AudioBuffer(new float[16000], 16000);
        Assert.Empty(pipeline.Transcribe(silence, "EN", token).Words);

        var original = new Transcript { Text = "东西", Words = [new Word { Text = "东西" }], Language = "zh" };
        var canonical = qwen.Align(silence, original, cancellationToken: token);
        foreach (string alias in new[] { "Chinese", "zh-Hant-TW", "ZHO" })
        {
            var aliasResult = qwen.Align(silence, original with { Language = alias }, cancellationToken: token);
            Assert.Equal(canonical.Words.Select(w => (w.Start, w.End, w.TimingMethod)), aliasResult.Words.Select(w => (w.Start, w.End, w.TimingMethod)));
        }
    }
}
