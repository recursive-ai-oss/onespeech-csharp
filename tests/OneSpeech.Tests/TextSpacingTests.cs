using Xunit;

namespace OneSpeech.Tests;

public sealed class TextSpacingTests
{
    [Fact]
    public void CleanTrimsSegmentAndWordsAndDropsBlankWords()
    {
        var segment = new Segment { Text = " Hello world", Words = [new Word { Text = " Hello" }, new Word { Text = " " }, new Word { Text = " world " }] };
        var clean = TextSpacing.Clean(segment);
        Assert.Equal("Hello world", clean.Text);
        Assert.Equal(["Hello", "world"], clean.Words.Select(w => w.Text));
        Assert.Same(clean, TextSpacing.Clean(clean));
    }

    [Fact]
    public void JoinSpacesLatinScriptsOnly()
    {
        Assert.Equal("Hello world again", TextSpacing.Join([" Hello", "world ", "", "again"]));
        Assert.Equal("你好世界", TextSpacing.Join(["你好", "世界"]));
        Assert.Equal("こんにちは。Hello", TextSpacing.Join(["こんにちは。", "Hello"]));
        Assert.Equal("สวัสดีครับ", TextSpacing.Join(["สวัสดี", "ครับ"]));
        Assert.Equal("안녕 하세요", TextSpacing.Join(["안녕", "하세요"]));
    }

    [Fact]
    public void TranscriptJsonWithLeadingSpacesStillReads()
    {
        var old = new Transcript { Text = " Hello world", Words = [new Word { Text = " Hello" }, new Word { Text = " world" }] };
        var back = Transcript.FromJson(old.ToJson());
        Assert.Equal(" Hello", back.Words[0].Text);
    }
}
