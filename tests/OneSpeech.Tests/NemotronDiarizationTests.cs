using OneSpeech.Diarization;
using Xunit;

namespace OneSpeech.Tests;

public sealed class NemotronDiarizationTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;
    [Fact]
    public void CatalogKeepsPublisherOnlyAndFixedLicenseProvenance()
    {
        var entry = Catalog.ModelCatalog.Default.Get("nvidia/Nemotron-3-Diarization");
        Assert.Equal(ModelTier.Tier2, entry.Tier); Assert.False(entry.IsDefault); Assert.False(entry.Gated);
        Assert.Empty(entry.ConvertedFiles); Assert.All(entry.Files, f => Assert.Null(f.MirrorUrl));
        Assert.Equal(8, entry.Capabilities.MaxSpeakers); Assert.False(entry.Capabilities.Streaming);
        Assert.Equal(Catalog.NemotronDiarizationAssets.Attribution, entry.Attribution);
        var wrong = new ModelInfo { Id = "wrong", Revision = entry.Revision, Tier = ModelTier.Tier1, Source = entry.Source, License = "MIT", Attribution = "wrong" };
        Assert.Throws<ModelIntegrityException>(() => NemotronDiarizer.Load(new Catalog.ResolvedModel("nonexistent", wrong), cancellationToken: Token));
    }
    [Fact]
    public void FullChunkBoundaryRetainsCenteredFeatureAndTrailingMaskParity()
    {
        float[] pcm = new float[36 * 16000];
        for (int i = 0; i < pcm.Length; i++) pcm[i] = .1f * MathF.Sin(i * .017f);
        var chunks = NemotronFeatureChunks.Read([new AudioBuffer(pcm, 16000)], Token).ToArray();
        Assert.Equal(2, chunks.Length);
        Assert.Equal(3601, chunks.Sum(c => c.OutputMelFrames));
        var full = new Audio.NemoLogMelSpectrogram().Extract(new AudioBuffer(pcm, 16000), normalize: false, cancellationToken: Token);
        int offset = 0;
        foreach (var chunk in chunks)
        {
            for (int t = 0; t < chunk.OutputMelFrames; t++) for (int m = 0; m < 128; m++)
                Assert.Equal(full[m * 3601 + offset + t], chunk.Stacked[t * 128 + m]);
            offset += chunk.OutputMelFrames;
        }
    }
    [Fact]
    public void TiedCacheSelectionRetainsOriginalActivityAcrossSpeakerChanges()
    {
        // Digests of 304 numerical frame IDs from the released Transformers 5.22.2
        // public cache API, Torch 2.15 CPU. No audio or biometric embeddings are used.
        string[] expected = ["FAA14044D3C6BC6B385957A1A0C4CDA3F1D9F69A12CE6586DF4AEC8994DA9EB3", "009211D1F7EBBD575350498DE99C18A896195F7701DF5CAC0A090C10E0F31D4A"];
        var cache = new ArrivalOrderSpeakerCache(new float[512]);
        for (int step = 0; step < 2; step++)
        {
            float[] chunk = new float[340 * 512];
            for (int t = 0; t < 340; t++) Array.Fill(chunk, step * 340 + t + 1, t * 512, 512);
            float[] window = cache.Window(chunk), probabilities = new float[window.Length / 512 * 64];
            Array.Fill(probabilities, 1 / (1 + MathF.Exp(8)));
            for (int t = 0; t < probabilities.Length / 8; t++) probabilities[t * 8 + step] = 1 / (1 + MathF.Exp(-8));
            cache.Update(window, probabilities, 340);
            float[] state = cache.Window([]);
            byte[] ids = new byte[304 * 4];
            for (int t = 0; t < 304; t++) System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(ids.AsSpan(t * 4), state[t * 512]);
            Assert.Equal(expected[step], System.Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(ids)));
        }
    }
    [Fact]
    public void CachePlacesUnusedSilenceAtTheTailAndKeepsABoundedFifo()
    {
        // Numerical frame identifiers, not voice or biometric speaker embeddings.
        var cache = new ArrivalOrderSpeakerCache(new float[512]);
        float[] window = new float[340 * 512], probabilities = new float[340 * 64];
        for (int t = 0; t < 340; t++)
        {
            Array.Fill(window, t + 1, t * 512, 512);
            for (int k = 0; k < 8; k++)
            {
                if (t is >= 200 and < 270) probabilities[t * 64 + k * 8] = .8f;
                if (t is >= 220 and < 224) probabilities[t * 64 + k * 8 + 1] = .8f;
            }
        }
        cache.Update(window, probabilities, 340);
        var state = cache.Window([]);
        Assert.Equal(304 * 512, state.Length); Assert.Equal(264, cache.CachedFrames); Assert.Equal(40, cache.FifoFrames);
        int[] exclusiveFrames = Enumerable.Range(200, 70).Where(t => t is < 220 or >= 224).ToArray();
        for (int t = 0; t < 66; t++) Assert.Equal(exclusiveFrames[t] + 1, state[t * 512]);
        Assert.Equal(0, state[66 * 512]);
        for (int t = 0; t < 4; t++) Assert.Equal(t + 221, state[(67 + t) * 512]);
        for (int t = 71; t < 264; t++) Assert.Equal(0, state[t * 512]);
        for (int t = 0; t < 40; t++) Assert.Equal(t + 301, state[(264 + t) * 512]);
    }
    [Fact]
    public void CacheExclusivityIsJointProbabilityAndPrecedesTheRecencyBonus()
    {
        float[] window = new float[340 * 512], probabilities = new float[340 * 64];
        for (int t = 0; t < 340; t++)
        {
            Array.Fill(window, t + 1, t * 512, 512);
            for (int k = 0; k < 8; k++)
            {
                if (t < 32) probabilities[t * 64 + k * 8] = .99f;
                if (t is >= 32 and < 64) probabilities[t * 64 + k * 8 + 1] = .99f;
                if (t == 280) { probabilities[t * 64 + k * 8] = .7f; probabilities[t * 64 + k * 8 + 1] = .3f; }
                if (t == 281) { probabilities[t * 64 + k * 8] = .65f; probabilities[t * 64 + k * 8 + 1] = .2f; }
            }
        }
        var cache = new ArrivalOrderSpeakerCache(new float[512]); cache.Update(window, probabilities, 340);
        var selected = cache.Window([]);
        Assert.Equal(282, selected[32 * 512]); // .65 * (1-.2) > .5, despite a margin below .5.
        Assert.DoesNotContain(281f, Enumerable.Range(0, 264).Select(t => selected[t * 512])); // .7 * (1-.3) < .5: bonus cannot rescue it.
    }
    [Fact]
    public void InitialCacheRecencyStartsAtTheNominalCacheBoundary()
    {
        // Distinct numerical frame identifiers make the recency boundary observable without a model.
        float[] window = new float[340 * 512], probabilities = new float[340 * 64];
        for (int t = 0; t < 340; t++)
        {
            Array.Fill(window, t + 1, t * 512, 512);
            float p = t < 264 ? .9f - t * .0004f : .79f;
            for (int k = 0; k < 8; k++) probabilities[t * 64 + k * 8] = p;
        }
        var cache = new ArrivalOrderSpeakerCache(new float[512]); cache.Update(window, probabilities, 340);
        var selected = cache.Window([]);
        for (int t = 0; t < 220; t++) Assert.Equal(t + 1, selected[t * 512]);
        for (int t = 220; t < 256; t++) Assert.Equal(t + 45, selected[t * 512]);
        for (int t = 256; t < 264; t++) Assert.Equal(0, selected[t * 512]);
    }
    [Fact]
    public void ActivityRetainsOverlapAndFollowsHigherProbabilityOnOneClock()
    {
        var segments = new NemotronActivitySegments();
        float[] probabilities = new float[4 * 8];
        probabilities[0] = .9f; probabilities[1] = .6f;
        probabilities[8] = .6f; probabilities[9] = .9f;
        probabilities[16] = .8f;
        segments.Append(probabilities, 4, 0, 640);
        var turns = segments.Complete();
        var coalesced = NemotronActivitySegments.Coalesce(turns);
        Assert.Equal(2, coalesced.Length);
        Assert.Equal(TimeSpan.FromMilliseconds(30), coalesced[0].End);
        Assert.Equal(TimeSpan.FromMilliseconds(20), coalesced[1].End);
        var transcript = new Transcript { Text = "one two gap", Words = [Word(0, 10, "one"), Word(10, 20, "two"), Word(30, 40, "gap")] };
        var merged = TranscriptDiarization.Merge(transcript, turns, Token);
        Assert.Equal("speaker_0", merged.Words[0].Speaker?.Id);
        Assert.Equal("speaker_1", merged.Words[1].Speaker?.Id);
        Assert.True(merged.Words[0].Overlap);
        Assert.True(merged.Words[1].Overlap);
        Assert.Equal(SpeakerLabel.Unassigned, merged.Words[2].Speaker);
    }
    private static Word Word(int start, int end, string text) => new() { Text = text, Start = TimeSpan.FromMilliseconds(start), End = TimeSpan.FromMilliseconds(end), Confidence = null, TimingMethod = TimingMethod.Native };

    [Theory]
    [InlineData(1)]
    [InlineData(399)]
    [InlineData(16000)]
    public void CenteredFeaturesAreIndependentOfPcmBlockBoundaries(int length)
    {
        float[] pcm = new float[length];
        for (int i = 0; i < length; i++) pcm[i] = .1f * MathF.Sin(i * .017f);
        var whole = NemotronFeatureChunks.Read([new AudioBuffer(pcm, 16000)], Token).Single();
        var split = NemotronFeatureChunks.Read(Enumerable.Range(0, (length + 112) / 113).Select(i => new AudioBuffer(pcm.AsMemory(i * 113, Math.Min(113, length - i * 113)), 16000)), Token).Single();
        Assert.Equal(whole.Stacked, split.Stacked);
        Assert.Equal(length / 160 + 1, whole.OutputMelFrames);
        Assert.Equal(length, whole.SampleEnd);
    }
    [Fact]
    public void EmptyWrongRateNonFiniteAndCancellationHaveExplicitBehavior()
    {
        Assert.Empty(NemotronFeatureChunks.Read([new AudioBuffer(Array.Empty<float>(), 16000)], Token));
        Assert.Throws<UnsupportedAudioException>(() => NemotronFeatureChunks.Read([new AudioBuffer(new float[100], 8000)], Token).ToArray());
        Assert.Throws<ArgumentException>(() => NemotronFeatureChunks.Read([new AudioBuffer(new[] { float.NaN }, 16000)], Token).ToArray());
        Assert.Throws<OperationCanceledException>(() => NemotronFeatureChunks.Read([], new CancellationToken(true)).ToArray());
    }
    [Fact]
    public void ActivityRejectsInvalidProbabilitiesAndClipsTheLastFrame()
    {
        var segments = new NemotronActivitySegments();
        float[] probabilities = new float[8]; probabilities[0] = .9f;
        segments.Append(probabilities, 1, 0, 1);
        Assert.Equal(TimeSpan.FromTicks(TimeSpan.TicksPerSecond / 16000), segments.Complete()[0].Turn.End);
        probabilities[1] = float.NaN;
        Assert.Throws<InvalidDataException>(() => new NemotronActivitySegments().Append(probabilities, 1, 0, 160));
    }
}
