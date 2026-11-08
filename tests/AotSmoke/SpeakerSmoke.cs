using OneSpeech;
using OneSpeech.Speakers;

internal static class SpeakerSmoke
{
    internal static void Run()
    {
        if (EmbeddingMath.Cosine([1, 0], [0, 1]) != 0) throw new InvalidOperationException("Cosine failed.");
        var mean = EmbeddingMath.Average(new ReadOnlyMemory<float>[] { new float[] { 1, 0 }, new float[] { 0, 1 } });
        if (EmbeddingMath.Cosine(mean, [1, 1]) < .999999) throw new InvalidOperationException("Average failed.");
        var calibration = LogisticScoreCalibration.Fit([-1, 1], [false, true], CancellationToken.None);
        if (calibration.Apply(1) <= .5) throw new InvalidOperationException("Calibration failed.");
        var cohort = new AdaptiveScoreNormalizer(new ReadOnlyMemory<float>[] { new float[] { 1, 0 }, new float[] { 0, 1 } });
        using var embedder = new BasisEmbedder();
        using var registry = new InMemorySpeakerRegistry(embedder, .5, cohort);
        var audio = new AudioBuffer(new float[] { 1, 0 }, 16000);
        registry.Enroll("basis", audio, CancellationToken.None);
        if (registry.Identify(audio, CancellationToken.None).SpeakerId != "basis" || registry.Verify(audio, audio, CancellationToken.None) < .5 || !registry.Remove("basis", CancellationToken.None))
            throw new InvalidOperationException("Speaker registry failed.");
    }

    // Algebraic test double, not a voice model or a biometric embedding.
    private sealed class BasisEmbedder : SpeakerEmbedder
    {
        public override ModelInfo Model { get; } = new() { Id = "basis", Source = "https://example.org", Revision = "none", Tier = ModelTier.Tier1, License = "MIT", Attribution = "Algebraic test double." };
        public override int Dimension => 2;
        public override int SampleRate => 16000;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            audio.Samples.Span.CopyTo(destination);
        }
        public override void Dispose() { }
    }
}
