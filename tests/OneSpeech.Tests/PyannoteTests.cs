using OneSpeech.Diarization;
using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class PyannoteTests
{
    private static readonly int[] Masks = [0, 1, 2, 4, 3, 5, 6];
    private static readonly int[] Forward = [0, 1];
    private static readonly int[] Reverse = [1, 0];
    [Fact]
    public void PowersetOrderAndFrameBoundariesAreExplicit()
    {
        Assert.Equal(Masks, PyannoteSegmenter.ClassMasks);
        Assert.Equal(0, PyannoteSegmenter.Boundary(0, 160000));
        Assert.Equal(630, PyannoteSegmenter.Boundary(1, 160000));
        Assert.Equal(160000, PyannoteSegmenter.Boundary(589, 160000));
        Assert.Equal(100, PyannoteSegmenter.Boundary(1, 100));
    }

    [Fact]
    public void ConstrainedAssignmentKeepsLocalSpeakersDistinctWithStableTies()
    {
        ReadOnlyMemory<float>[] points = [new float[] { 1, 0 }, new float[] { 1, 0 }];
        float[][] centers = [[1, 0], [0, 1]];
        Assert.Equal(Forward, PyannoteDiarizer.Assign(points, centers));
        Assert.Equal(Reverse, PyannoteDiarizer.Assign([new float[] { 0, 1 }, new float[] { 1, 0 }], centers));
        Assert.Throws<ArgumentException>(() => PyannoteDiarizer.Assign(points, [[1, 0]]));
    }

    [Fact]
    public void ActivityCreatesOverlappingTurnsOnOneClockAndClipsPadding()
    {
        int[] activity = new int[589]; activity[0] = 3; activity[1] = 1;
        var turns = PyannoteDiarizer.FrameTurns(160000, 160000, activity, [0, 1], [0, 1]);
        Assert.Equal(2, turns.Length);
        Assert.All(turns, turn => Assert.Equal(TimeSpan.FromSeconds(10), turn.Start));
        Assert.Equal(TimeSpan.FromSeconds(10 + 900 / 16000d), turns[0].End);
        Assert.Equal(TimeSpan.FromSeconds(10 + 630 / 16000d), turns[1].End);
        Assert.NotEqual(turns[0].Speaker, turns[1].Speaker);
        var clipped = PyannoteDiarizer.FrameTurns(0, 100, activity, [0, 1], [0, 1]);
        Assert.All(clipped, turn => Assert.Equal(TimeSpan.FromSeconds(100 / 16000d), turn.End));
        Assert.Empty(PyannoteDiarizer.FrameTurns(0, 160000, new int[589], [0, 1], [0, 1]));
    }

    [Fact]
    public void WrongLocalWeightsAreRefusedBeforeAnyCacheWrites()
    {
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllBytes(Path.Combine(root, "pytorch_model.bin"), [0, 1, 2]);
            string cache = Path.Combine(root, "cache");
            Assert.Throws<ModelIntegrityException>(() => PyannoteSegmentation.LoadLocal(root, cache, cancellationToken: TestContext.Current.CancellationToken));
            Assert.False(Directory.Exists(cache));
            using var canceled = new CancellationTokenSource(); canceled.Cancel();
            Assert.Throws<OperationCanceledException>(() => PyannoteSegmentation.LoadLocal(root, cache, cancellationToken: canceled.Token));
            Assert.False(Directory.Exists(cache));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void LocalSnapshotStaysReadOnlyAndConversionsStayInCache()
    {
        string? snapshot = Environment.GetEnvironmentVariable("ONESPEECH_PYANNOTE_SNAPSHOT");
        Assert.SkipWhen(snapshot is null, "Set ONESPEECH_PYANNOTE_SNAPSHOT to the authorized pinned publisher snapshot.");
        string cache = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        string[] before = Directory.GetFiles(snapshot!, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToArray();
        try
        {
            using var model = PyannoteSegmentation.LoadLocal(snapshot!, cache, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(ModelTier.Tier2, model.Model.Tier); Assert.True(model.Model.Gated);
            Assert.Equal("MIT", model.Model.License);
            Assert.Contains("e66f3d3b9eb0873085418a7b813d3b369bf160bb", model.Model.Attribution);
            Assert.Equal(2, model.Model.Files.Length);
            Assert.Single(Directory.GetFiles(cache, "*.onemodel", SearchOption.AllDirectories));
            Assert.Empty(Directory.GetFiles(cache, "*.bin", SearchOption.AllDirectories));
            var zero = new AudioBuffer(new float[160000], 16000);
            var probabilities = model.GetFrameLogProbabilities(zero, TestContext.Current.CancellationToken);
            Assert.Equal(589 * 7, probabilities.Length);
            for (int f = 0; f < 589; f++) Assert.InRange(Enumerable.Range(0, 7).Sum(c => Math.Exp(probabilities[f * 7 + c])), 0.99999, 1.00001);
            Assert.Throws<ArgumentException>(() => model.GetFrameLogProbabilities(new AudioBuffer(new float[1600], 16000), TestContext.Current.CancellationToken));
            Assert.Throws<ArgumentException>(() => model.GetFrameLogProbabilities(new AudioBuffer(new float[160000], 8000), TestContext.Current.CancellationToken));
            Assert.Equal(before, Directory.GetFiles(snapshot!, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToArray());
            model.Dispose(); model.Dispose();
            Assert.Throws<ObjectDisposedException>(() => model.GetFrameLogProbabilities(zero, TestContext.Current.CancellationToken));
        }
        finally { if (Directory.Exists(cache)) Directory.Delete(cache, recursive: true); }
    }
}
