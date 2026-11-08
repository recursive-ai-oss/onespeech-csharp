using OneSpeech.Diarization;
using OneSpeech.Pipeline;
using OneSpeech.Speakers;
using Xunit;

namespace OneSpeech.Tests;

[CollectionDefinition("Speaker statics", DisableParallelization = true)]
public sealed class SpeakerStaticGroup;

[Collection("Speaker statics")]
public sealed class SpeakerStaticTests
{
    [Fact]
    public async Task ValidationCancellationAndClearDoNotLoadModels()
    {
        Speech.ClearSpeakers(TestContext.Current.CancellationToken); Speech.ClearSpeakers(TestContext.Current.CancellationToken);
        Assert.Throws<ArgumentException>(() => Speech.Enroll(" ", "missing", cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentNullException>(() => Speech.Enroll(null!, "missing", cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<CapabilityNotSupportedException>(() => Speech.Identify("missing", model: "whisper-tiny", cancellationToken: TestContext.Current.CancellationToken));
        using var cancel = new CancellationTokenSource(); cancel.Cancel();
        Assert.Throws<OperationCanceledException>(() => Speech.Identify("missing", cancellationToken: cancel.Token));
        Assert.Throws<OperationCanceledException>(() => Speech.ClearSpeakers(cancel.Token));
        await Assert.ThrowsAsync<OperationCanceledException>(async () => await Speech.IdentifyAsync(new MemoryStream(), cancellationToken: cancel.Token));
        await Assert.ThrowsAsync<ArgumentException>(async () => await Speech.EnrollAsync(" ", new MemoryStream(), cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void NamedClustersPreserveDocumentIdsAndUnknowns()
    {
        using var embedder = new BasisEmbedder();
        using var registry = new InMemorySpeakerRegistry(embedder, .9);
        registry.Enroll("Alice", new AudioBuffer(new float[] { 1, 0 }, 16000), CancellationToken.None);
        var diarizer = new ClusteringDiarizer(embedder, new WholeVad(), new ClusteringOptions { Resegment = false })
        {
            IdentifyCluster = registry.IdentifyEmbedding
        };
        var turns = diarizer.Diarize(new AudioBuffer(Enumerable.Repeat(1f, 16000).ToArray(), 16000), CancellationToken.None);
        Assert.NotEmpty(turns);
        Assert.All(turns, t => { Assert.Equal("speaker-1", t.Speaker.Id); Assert.Equal("Alice", t.Speaker.Name); });
        var transcript = new Transcript { Text = "hello", Words = [new Word { Text = "hello", Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(.5) }] };
        var named = TranscriptDiarization.Merge(transcript, turns.Select(t => new ActiveTurn { Turn = t }), CancellationToken.None);
        Assert.Equal("Alice", named.Words[0].Speaker!.Name);
        Assert.Null(transcript.Words[0].Speaker);
        var unknown = diarizer.Diarize(new AudioBuffer(new float[16000], 16000), CancellationToken.None);
        Assert.All(unknown, t => Assert.Null(t.Speaker.Name));
    }

    [Fact]
    public async Task PublicSpeakersRoundTripAndNameTranscription()
    {
        string? corpus = Environment.GetEnvironmentVariable("ONESPEECH_SPEAKER_CORPUS");
        if (corpus is null) { Assert.Skip("Set ONESPEECH_SPEAKER_CORPUS to the public LibriSpeech corpus root and provide the model cache."); return; }
        string Find(string id) => Directory.EnumerateFiles(corpus, id + ".flac", SearchOption.AllDirectories).Single();
        // Holdout identities: neither is in the threshold-development speaker partition.
        string enroll = Find("3575-170457-0000"), query = Find("3575-170457-0001"), other = Find("4077-13751-0000");
        Speech.ClearSpeakers(TestContext.Current.CancellationToken);
        try
        {
            Speech.Enroll("Reader", enroll, cancellationToken: TestContext.Current.CancellationToken);
            var match = Speech.Identify(query, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal("Reader", match.SpeakerId);
            Assert.True(match.Score >= SpeakerServices.EnglishThreshold);
            Assert.Null(Speech.Identify(other, cancellationToken: TestContext.Current.CancellationToken).SpeakerId);
            Assert.True(Speech.Verify(enroll, query, cancellationToken: TestContext.Current.CancellationToken) > Speech.Verify(enroll, other, cancellationToken: TestContext.Current.CancellationToken));
            await using var enrollment = File.OpenRead(enroll);
            await using var input = File.OpenRead(query);
            await Speech.EnrollAsync("Reader", enrollment, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal("Reader", (await Speech.IdentifyAsync(input, cancellationToken: TestContext.Current.CancellationToken)).SpeakerId);
            Assert.True(enrollment.CanRead); Assert.True(input.CanRead);
            await using var a = File.OpenRead(enroll);
            await using var b = File.OpenRead(query);
            Assert.Equal(Speech.Verify(enroll, query, cancellationToken: TestContext.Current.CancellationToken), await Speech.VerifyAsync(a, b, cancellationToken: TestContext.Current.CancellationToken), 6);
            var transcript = Speech.Transcribe(query, language: "en", model: "openai/whisper-tiny", cancellationToken: TestContext.Current.CancellationToken);
            Assert.NotEmpty(transcript.Turns);
            Assert.All(transcript.Turns, t => Assert.Equal("Reader", t.Speaker.Name));
            Assert.Contains(transcript.Words, w => w.Speaker?.Name == "Reader");
            Console.WriteLine($"known score={match.Score:R}; unknown score={Speech.Identify(other, cancellationToken: TestContext.Current.CancellationToken).Score:R}; named turns={transcript.Turns.Length}");
            Speech.ClearSpeakers(TestContext.Current.CancellationToken);
            Assert.Null(Speech.Identify(query, cancellationToken: TestContext.Current.CancellationToken).SpeakerId);
            Assert.Equal(double.NegativeInfinity, Speech.Identify(query, cancellationToken: TestContext.Current.CancellationToken).Score);
        }
        finally { Speech.ClearSpeakers(TestContext.Current.CancellationToken); }
    }

    // Algebraic basis vectors only, not voice embeddings.
    private sealed class BasisEmbedder : SpeakerEmbedder
    {
        public override ModelInfo Model { get; } = new() { Id = "basis", Source = "https://example.org", Revision = "none", Tier = ModelTier.Tier1, License = "MIT", Attribution = "Algebraic test double" };
        public override int Dimension => 2;
        public override int SampleRate => 16000;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            bool first = audio.Samples.Span[0] != 0;
            destination[0] = first ? 1 : 0; destination[1] = first ? 0 : 1;
        }
        public override void Dispose() { }
    }
    private sealed class WholeVad : VoiceActivityDetector
    {
        public override ModelInfo Model { get; } = new BasisEmbedder().Model;
        public override System.Collections.Immutable.ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken)
            => [new SpeechInterval { Start = TimeSpan.Zero, End = audio.Duration }];
    }
}
