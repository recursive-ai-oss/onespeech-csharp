using OneSpeech.Media;
using RecursiveAI.OneMedia;
using RecursiveAI.OneMedia.Captions;
using Xunit;

namespace OneSpeech.Media.Tests;

public sealed class CaptionTests
{
    internal static Transcript Example()
    {
        Word[] words = [new() { Text = "Hello", Start = TimeSpan.FromSeconds(1), End = TimeSpan.FromSeconds(1.4), Speaker = new() { Id = "alice" } }, new() { Text = " world", Start = TimeSpan.FromSeconds(1.5), End = TimeSpan.FromSeconds(2), Speaker = new() { Id = "alice" } }];
        return new() { Text = "Hello world", Words = [.. words], Segments = [new() { Text = "Hello world", Start = words[0].Start, End = words[1].End, Words = [.. words] }], Language = "en" };
    }

    [Theory]
    [InlineData("srt")]
    [InlineData("vtt")]
    [InlineData("ttml")]
    [InlineData("ass")]
    public void WritersRoundTripTextAndRepresentableMetadata(string name)
    {
        var transcript = Example();
        string output = name switch { "srt" => transcript.ToSrt(TestContext.Current.CancellationToken), "vtt" => transcript.ToVtt(TestContext.Current.CancellationToken), "ttml" => transcript.ToTtml(TestContext.Current.CancellationToken), _ => transcript.ToAss(TestContext.Current.CancellationToken) };
        var doc = SubtitleFormats.Find(name)!.ReadText(output);
        var cue = Assert.Single(doc.Cues);
        Assert.Equal("Hello world", cue.Text);
        Assert.Equal(TimeSpan.FromSeconds(1), cue.Start.ToTimeSpan());
        Assert.Equal(TimeSpan.FromSeconds(2), cue.End.ToTimeSpan());
        if (name == "srt") Assert.Null(cue.Speaker);
        else Assert.True(cue.Speaker == "alice" || cue.Spans.Any(s => s.Voice == "alice"));
        if (name is "vtt" or "ttml")
        {
            Assert.Contains(cue.Spans, s => s.WordStart?.ToTimeSpan() == TimeSpan.FromSeconds(1.5));
            var imported = doc.ToTranscript(TestContext.Current.CancellationToken);
            Assert.Contains(imported.Words, w => w.Start == TimeSpan.FromSeconds(1.5));
            Assert.All(imported.Words, w => Assert.Null(w.Confidence));
        }
    }

    [Fact]
    public void UntimedWordsDoNotReceiveTimesAndUnknownCuesAreNotEstimated()
    {
        var t = Example();
        var words = t.Words.SetItem(1, t.Words[1] with { Start = null, End = null });
        t = t with { Words = words, Segments = [t.Segments[0] with { Words = words }] };
        var cue = Assert.Single(t.ToSubtitles(TestContext.Current.CancellationToken).Cues);
        Assert.Null(cue.Spans[1].WordStart);
        Assert.Null(cue.Spans[1].WordEnd);
        var restored = t.ToSubtitles(TestContext.Current.CancellationToken).ToTranscript(TestContext.Current.CancellationToken);
        Assert.Equal(2, restored.Words.Length);
        Assert.Null(restored.Words[1].Start);
        Assert.Null(restored.Words[1].End);
        Assert.Equal(TimingMethod.None, restored.Words[1].TimingMethod);
        var unknown = new Transcript { Text = "unknown", Words = [new() { Text = "unknown" }] }.ToSubtitles(TestContext.Current.CancellationToken);
        Assert.False(Assert.Single(unknown.Cues).Start.IsKnown);
        Assert.False(unknown.Cues[0].End.IsKnown);
        var untimed = unknown.ToTranscript(TestContext.Current.CancellationToken);
        Assert.Throws<CapabilityNotSupportedException>(() => untimed.ToSrt(TestContext.Current.CancellationToken));
        Assert.Throws<CapabilityNotSupportedException>(() => untimed.ToVtt(TestContext.Current.CancellationToken));
        Assert.Throws<CapabilityNotSupportedException>(() => untimed.ToAss(TestContext.Current.CancellationToken));
        Assert.Throws<CapabilityNotSupportedException>(() => untimed.ToTtml(TestContext.Current.CancellationToken));
        Assert.Empty(untimed.EncodeCaptionTrack(cancellationToken: TestContext.Current.CancellationToken).Packets);
        Assert.Empty(new Transcript { Text = "" }.ToSrt(TestContext.Current.CancellationToken));
    }

    [Fact]
    public void TurnSpeakersAndExclusiveDisplayKeepOriginalWordTimes()
    {
        var t = Example();
        t = t with { Turns = [new() { Text = t.Text, Speaker = new() { Id = "bob" }, Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(3) }] };
        Assert.Equal("bob", Assert.Single(t.ToSubtitles(TestContext.Current.CancellationToken).Cues).Speaker);
        var later = t.Segments[0] with { Words = [new() { Text = "other", Start = TimeSpan.FromSeconds(1.8), End = TimeSpan.FromSeconds(3) }] };
        var cues = (t with { Turns = [], Segments = [t.Segments[0], later] }).ToSubtitles(TestContext.Current.CancellationToken).Cues;
        Assert.Equal(2, cues.Count);
        Assert.Equal(cues[1].Start, cues[0].End);
        Assert.Equal(TimeSpan.FromSeconds(2), cues[0].Spans[1].WordEnd!.Value.ToTimeSpan());
    }

    [Fact]
    public void LongCaptionsSplitAndWrapWithoutSplittingWords()
    {
        var words = Enumerable.Range(0, 30).Select(i => new Word { Text = "caption", Start = TimeSpan.FromSeconds(i / 3d), End = TimeSpan.FromSeconds((i + 1) / 3d) }).ToArray();
        var doc = new Transcript { Text = "", Words = [.. words] }.ToSubtitles(TestContext.Current.CancellationToken);
        Assert.True(doc.Cues.Count > 1);
        Assert.All(doc.Cues, cue => Assert.All(cue.Text.Split('\n'), line => Assert.True(line.Length <= 42)));
        Assert.Equal(30, doc.Cues.Sum(c => c.Spans.Count));
        Assert.All(doc.Cues, cue => Assert.True(cue.Text.Length <= 84));
    }

    [Fact]
    public void CancellationAndInvalidImportedRangesAreRefused()
    {
        using var cts = new CancellationTokenSource(); cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => Example().ToSubtitles(cts.Token));
        var bad = new SubtitleDocument { Cues = SubtitleList.Of(new SubtitleCue { Text = "x", Spans = SubtitleList.Of(new SubtitleSpan(9, 1)) }) };
        Assert.Throws<ArgumentException>(() => bad.ToTranscript(TestContext.Current.CancellationToken));
    }
}

public sealed class CaptionSpacingTests
{
    private static Transcript Make(string language, params string[] words)
    {
        var list = words.Select((w, i) => new Word { Text = w, Start = TimeSpan.FromSeconds(i), End = TimeSpan.FromSeconds(i + .9) }).ToArray();
        return new() { Text = "", Language = language, Words = [.. list], Segments = [new() { Text = "", Start = list[0].Start, End = list[^1].End, Words = [.. list] }] };
    }

    [Fact]
    public void UnspacedWordsGetOneSpaceInCues()
    {
        var doc = SubtitleFormats.Find("srt")!.ReadText(Make("en", "Hello", "world").ToSrt(TestContext.Current.CancellationToken));
        Assert.Equal("Hello world", Assert.Single(doc.Cues).Text);
    }

    [Fact]
    public void ChineseWordsAreNotSpaced()
    {
        var doc = SubtitleFormats.Find("srt")!.ReadText(Make("zh", "你好", "世界").ToSrt(TestContext.Current.CancellationToken));
        Assert.Equal("你好世界", Assert.Single(doc.Cues).Text);
    }
}
