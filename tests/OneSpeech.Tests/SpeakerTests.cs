using OneSpeech.Speakers;
using Xunit;

namespace OneSpeech.Tests;

// The vectors below are algebraic basis vectors, not biometric embeddings or generated voices.
public sealed class SpeakerTests
{
    [Fact]
    public void CosineAndAverageAreScaleInvariant()
    {
        Assert.Equal(1, EmbeddingMath.Cosine([3, 4], [6, 8]), 12);
        Assert.Equal(-1, EmbeddingMath.Cosine([3, 4], [-6, -8]), 12);
        Assert.Equal(0, EmbeddingMath.Cosine([1, 0], [0, 1]));
        var mean = EmbeddingMath.Average(new ReadOnlyMemory<float>[] { new float[] { 100, 0 }, new float[] { 0, 1 } });
        Assert.Equal(1, EmbeddingMath.Cosine(mean, [1, 1]), 12);
        Assert.Equal(1, EmbeddingMath.Cosine([float.MaxValue, float.MaxValue], [1, 1]), 12);
        Assert.Equal(1, EmbeddingMath.Cosine([float.Epsilon, 0], [1, 0]), 12);
        float[] vector = [3, 4];
        EmbeddingMath.Normalize(vector, vector);
        Assert.Equal(.6f, vector[0]);
        Assert.Equal(.8f, vector[1]);
    }

    [Fact]
    public void InvalidVectorsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Cosine([], []));
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Cosine([0, 0], [1, 0]));
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Cosine([1], [1, 0]));
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Cosine([float.NaN], [1]));
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Normalize([float.PositiveInfinity], new float[1]));
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Average([]));
        Assert.Throws<ArgumentException>(() => EmbeddingMath.Average(new ReadOnlyMemory<float>[] { new float[] { 1 }, new float[] { -1 } }));
    }

    [Fact]
    public void CohortNormalizationUsesTopScoresAndCopiesInputs()
    {
        float[] first = [1, 0];
        var normalizer = new AdaptiveScoreNormalizer(new ReadOnlyMemory<float>[] { first, new float[] { 0, 1 }, new float[] { -1, 0 } }, topCount: 2);
        Array.Clear(first);
        Assert.Equal(1, normalizer.Score([1, 0], [1, 0], CancellationToken.None), 12);
        Assert.Equal(-1, normalizer.Score([1, 0], [0, 1], CancellationToken.None), 12);
        Assert.Equal(normalizer.Score([1, 0], [1, 1], CancellationToken.None), normalizer.Score([1, 1], [1, 0], CancellationToken.None));
        var identical = new AdaptiveScoreNormalizer(new ReadOnlyMemory<float>[] { new float[] { 1, 0 }, new float[] { 1, 0 } });
        Assert.True(double.IsFinite(identical.Score([1, 0], [0, 1], CancellationToken.None)));
        Assert.Throws<OperationCanceledException>(() => normalizer.Score([1, 0], [0, 1], new CancellationToken(true)));
    }

    [Fact]
    public void CalibrationFitIsDeterministicAndHandlesBothSigns()
    {
        double[] scores = [-1, -.5, .5, 1];
        bool[] labels = [false, false, true, true];
        var first = LogisticScoreCalibration.Fit(scores, labels, CancellationToken.None);
        var second = LogisticScoreCalibration.Fit(scores, labels, CancellationToken.None);
        Assert.Equal(first.Apply(.3), second.Apply(.3));
        Assert.Equal(.5, first.Apply(0), 12);
        Assert.True(first.Apply(1) > .95);
        Assert.True(first.Apply(-1) < .05);
        var reversed = LogisticScoreCalibration.Fit(scores, [true, true, false, false], CancellationToken.None);
        Assert.True(reversed.Apply(1) < .05);
        var constant = LogisticScoreCalibration.Fit([2, 2], [false, true], CancellationToken.None);
        Assert.Equal(.5, constant.Apply(2));
        Assert.Throws<ArgumentException>(() => LogisticScoreCalibration.Fit(scores, [true, true, true, true], CancellationToken.None));
        Assert.Throws<ArgumentException>(() => LogisticScoreCalibration.Fit([double.NaN, 0], [false, true], CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => first.Apply(double.PositiveInfinity));
        Assert.Throws<OperationCanceledException>(() => LogisticScoreCalibration.Fit(scores, labels, new CancellationToken(true)));
    }

    [Fact]
    public void RegistryEnrollIdentifyReplaceUnknownThresholdAndRemove()
    {
        using var embedder = new BasisEmbedder();
        using var registry = new InMemorySpeakerRegistry(embedder, .9);
        var a = Audio(1, 0);
        var b = Audio(0, 1);
        Assert.Null(registry.Identify(a, CancellationToken.None).SpeakerId);
        Assert.Equal(double.NegativeInfinity, registry.Identify(a, CancellationToken.None).Score);
        registry.Enroll("a", a, CancellationToken.None);
        Assert.Equal("a", registry.Identify(a, CancellationToken.None).SpeakerId);
        Assert.Null(registry.Identify(b, CancellationToken.None).SpeakerId);
        Assert.Equal(1, registry.Verify(a, a, CancellationToken.None));
        Assert.Equal(0, registry.Verify(a, b, CancellationToken.None));
        registry.Enroll("a", b, CancellationToken.None);
        Assert.Equal("a", registry.Identify(b, CancellationToken.None).SpeakerId);
        Assert.Null(registry.Identify(a, CancellationToken.None).SpeakerId);
        Assert.True(registry.Remove("a", CancellationToken.None));
        Assert.False(registry.Remove("a", CancellationToken.None));
        registry.Enroll("z", a, CancellationToken.None);
        registry.Enroll("a", a, CancellationToken.None);
        Assert.Equal("a", registry.Identify(a, CancellationToken.None).SpeakerId);
        using var inclusive = new InMemorySpeakerRegistry(embedder, 1);
        inclusive.Enroll("a", a, CancellationToken.None);
        Assert.Equal("a", inclusive.Identify(a, CancellationToken.None).SpeakerId);
    }

    [Fact]
    public void RegistryAveragesUtterancesAndKeepsEnrollmentOnFailure()
    {
        using var embedder = new BasisEmbedder();
        using var registry = new InMemorySpeakerRegistry(embedder, .99);
        registry.Enroll("mean", new[] { Audio(10, 0), Audio(0, 1) }, CancellationToken.None);
        Assert.Equal("mean", registry.Identify(Audio(1, 1), CancellationToken.None).SpeakerId);
        Assert.Throws<ArgumentException>(() => registry.Enroll("mean", Audio(0, 0), CancellationToken.None));
        Assert.Equal("mean", registry.Identify(Audio(1, 1), CancellationToken.None).SpeakerId);
        Assert.Throws<OperationCanceledException>(() => registry.Remove("mean", new CancellationToken(true)));
        Assert.Equal("mean", registry.Identify(Audio(1, 1), CancellationToken.None).SpeakerId);
        Assert.Throws<ArgumentException>(() => registry.Enroll(" ", Audio(1, 0), CancellationToken.None));
        Assert.Throws<ArgumentException>(() => registry.Verify(new AudioBuffer(new float[4], 8000), Audio(1, 0), CancellationToken.None));
        registry.Dispose();
        Assert.Throws<ObjectDisposedException>(() => registry.Identify(Audio(1, 0), CancellationToken.None));
        Assert.False(embedder.Disposed);
    }

    [Fact]
    public void FailedBatchEnrollmentIsAtomicAndConfigurationIsValidated()
    {
        using var embedder = new BasisEmbedder();
        using var registry = new InMemorySpeakerRegistry(embedder, .99);
        registry.Enroll("saved", Audio(1, 0), CancellationToken.None);
        using var cancellation = new CancellationTokenSource();
        embedder.AfterEmbed = cancellation.Cancel;
        Assert.Throws<OperationCanceledException>(() => registry.Enroll("saved", new[] { Audio(0, 1), Audio(0, 1) }, cancellation.Token));
        embedder.AfterEmbed = null;
        Assert.Equal("saved", registry.Identify(Audio(1, 0), CancellationToken.None).SpeakerId);
        Assert.Throws<ArgumentOutOfRangeException>(() => new InMemorySpeakerRegistry(embedder, 1.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new InMemorySpeakerRegistry(embedder, double.NaN));
        var wrongCohort = new AdaptiveScoreNormalizer(new ReadOnlyMemory<float>[] { new float[] { 1 }, new float[] { -1 } });
        Assert.Throws<ArgumentException>(() => new InMemorySpeakerRegistry(embedder, .5, wrongCohort));
        Assert.Throws<ArgumentException>(() => registry.Enroll("saved", Array.Empty<AudioBuffer>(), CancellationToken.None));
    }

    private static AudioBuffer Audio(float a, float b) => new(new float[] { a, b }, 16000);

    private sealed class BasisEmbedder : SpeakerEmbedder
    {
        public bool Disposed { get; private set; }
        public Action? AfterEmbed { get; set; }
        public override ModelInfo Model { get; } = new() { Id = "algebra", Source = "https://example.org", Revision = "none", Tier = ModelTier.Tier1, License = "MIT", Attribution = "Algebraic test double; not a voice model." };
        public override int Dimension => 2;
        public override int SampleRate => 16000;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            ObjectDisposedException.ThrowIf(Disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            audio.Samples.Span.CopyTo(destination);
            AfterEmbed?.Invoke();
        }
        public override void Dispose() => Disposed = true;
    }
}
