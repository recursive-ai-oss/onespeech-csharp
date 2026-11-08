using System.Text;
using OneSpeech.Evaluation;
using Xunit;

namespace OneSpeech.Eval.Tests;

public sealed class DiarizationMetricsTests
{
    private static DiarizationSegment S(string speaker, double start, double end) => new("call", "1", speaker, start, end);

    [Fact]
    public void MissFalseAlarmAndNoCollar()
    {
        var score = DiarizationMetrics.Score([S("a", 0, 2)], [S("b", 1, 3)], TestContext.Current.CancellationToken);
        Assert.Equal(2, score.ReferenceSeconds);
        Assert.Equal(1, score.MissSeconds);
        Assert.Equal(1, score.FalseAlarmSeconds);
        Assert.Equal(0, score.ConfusionSeconds);
        Assert.Equal(1, score.Der);
        Assert.Equal(2.0 / 3, score.Jer!.Value, 12);
        var boundary = DiarizationMetrics.Score([S("a", 0, 1)], [S("b", 0.001, 1)], TestContext.Current.CancellationToken);
        Assert.Equal(0.001, boundary.Der!.Value, 12);
    }

    [Fact]
    public void OneGlobalMappingCountsConfusion()
    {
        var score = DiarizationMetrics.Score([S("a", 0, 1), S("b", 1, 2)], [S("x", 0, 2)], TestContext.Current.CancellationToken);
        Assert.Equal(1, score.ConfusionSeconds);
        Assert.Equal(0.5, score.Der);
        Assert.Equal(0.75, score.Jer);
    }

    [Fact]
    public void OverlapIsSpeakerTimeNotWallTimeAndDuplicatesAreUnioned()
    {
        var score = DiarizationMetrics.Score([S("a", 0, 2), S("b", 1, 2), S("a", 0.5, 1.5)], [S("x", 0, 2)], TestContext.Current.CancellationToken);
        Assert.Equal(3, score.ReferenceSeconds);
        Assert.Equal(1, score.MissSeconds);
        Assert.Equal(0, score.ConfusionSeconds);
        Assert.Equal(1.0 / 3, score.Der!.Value, 12);
        Assert.Equal(0.5, score.Jer);
        var perfect = DiarizationMetrics.Score([S("a", 0, 2), S("b", 1, 2)], [S("y", 1, 2), S("x", 0, 2)], TestContext.Current.CancellationToken);
        Assert.Equal(0, perfect.Der); Assert.Equal(0, perfect.Jer);
    }

    [Fact]
    public void HungarianMappingIsNotGreedyAndJerHasItsOwnMapping()
    {
        // Intersection matrix [[6,5],[5,0]]. Off-diagonal is optimal for DER (10 rather than 6).
        var score = DiarizationMetrics.Score([S("a", 0, 11), S("b", 11, 16)],
            [S("x", 0, 6), S("y", 6, 11), S("x", 11, 16)], TestContext.Current.CancellationToken);
        Assert.Equal(6, score.ConfusionSeconds);
        Assert.Equal(6.0 / 16, score.Der);
    }

    [Fact]
    public void EmptyAndMissingRecordingsAreNotSilentlyDiscarded()
    {
        var score = DiarizationMetrics.Score([], [S("x", 0, 1)], TestContext.Current.CancellationToken);
        Assert.Null(score.Der); Assert.Null(score.Jer); Assert.Equal(1, score.FalseAlarmSeconds);
        score = DiarizationMetrics.Score([S("a", 0, 1)], [], TestContext.Current.CancellationToken);
        Assert.Equal(1, score.Der); Assert.Equal(1, score.Jer);
        score = DiarizationMetrics.Score([S("a", 0, 1)], [S("x", 0, 1) with { Recording = "extra" }], TestContext.Current.CancellationToken);
        Assert.Equal(2, score.Der);
    }

    [Theory]
    [InlineData("SPEAKER call 1 0 1 <NA> <NA> a <NA> <NA>", true)]
    [InlineData("# comment\n\nSPEAKER call 1 0 1 <NA> <NA> a <NA> <NA>\r\n", true)]
    [InlineData("SPEAKER call 1 NaN 1 <NA> <NA> a <NA> <NA>", false)]
    [InlineData("SPEAKER call 1 0 -1 <NA> <NA> a <NA> <NA>", false)]
    [InlineData("SPEAKER call 1 1e308 1 <NA> <NA> a <NA> <NA>", false)]
    [InlineData("SPEAKER call 0 0 1 <NA> <NA> a <NA> <NA>", false)]
    [InlineData("SPEAKER call 1 0 1 <NA> <NA> a <NA>", false)]
    public void ReaderChecksHostileTimesAndFields(string text, bool valid)
    {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
        if (valid) Assert.Single(Rttm.Read(stream));
        else Assert.Throws<InvalidDataException>(() => Rttm.Read(stream));
    }

    [Fact]
    public void ReaderBoundsAndUtf8()
    {
        using var huge = new MemoryStream(Encoding.UTF8.GetBytes(new string('x', 16385)));
        Assert.Throws<InvalidDataException>(() => Rttm.Read(huge));
        using var invalid = new MemoryStream([0xff, 0xfe, 0xff]);
        Assert.Throws<InvalidDataException>(() => Rttm.Read(invalid));
        var transcript = new Transcript { Text = "", Turns = [new Turn { Speaker = new SpeakerLabel { Id = "a" }, Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(1), Text = "" }] };
        Assert.Equal("meeting", Assert.Single(Rttm.FromTranscript(transcript, "meeting")).Recording);
    }
}
