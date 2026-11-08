using OneSpeech.Pipeline;
using Xunit;

namespace OneSpeech.Tests;

public sealed class DiarizationDefaultsTests
{
    [Fact]
    public void DiarizationActivityDoesNotChangeAsrDefaults()
    {
        var asr = new OneSpeech.Vad.SileroVadOptions();
        Assert.Equal(0.5f, asr.Threshold);
        Assert.Equal(TimeSpan.FromMilliseconds(100), asr.MinimumSilence);
        Assert.Equal(TimeSpan.FromMilliseconds(30), asr.SpeechPadding);
        var activity = SpeechServices.DiarizationActivity;
        Assert.Equal(0.2f, activity.Threshold);
        Assert.Equal(0.2f - 0.15f, activity.NegativeThreshold);
        Assert.Equal(TimeSpan.FromMilliseconds(250), activity.MinimumSpeech);
        Assert.Equal(TimeSpan.FromMilliseconds(250), activity.MinimumSilence);
        Assert.Equal(TimeSpan.FromMilliseconds(150), activity.SpeechPadding);
    }

    [Theory]
    [InlineData("zh")]
    [InlineData("zh-CN")]
    [InlineData("ZH-TW")]
    [InlineData("Chinese")]
    public void ChineseUsesChineseEmbedder(string language) =>
        Assert.Equal("iic/speech_campplus_sv_zh-cn_16k-common", SpeechServices.EmbedderId(language));

    [Theory]
    [InlineData(null)]
    [InlineData("en")]
    [InlineData("de")]
    public void OtherLanguagesUseEnglishEmbedder(string? language) =>
        Assert.Equal("iic/speech_campplus_sv_en_voxceleb_16k", SpeechServices.EmbedderId(language));

    [Theory]
    [InlineData(2798640000L)]
    [InlineData(1L)]
    [InlineData(2798640001L)]
    [InlineData(long.MaxValue)]
    public void JsonPreservesEveryTick(long ticks)
    {
        var transcript = new Transcript { Text = "", Duration = TimeSpan.FromTicks(ticks) };
        Assert.Equal(ticks, Transcript.FromJson(transcript.ToJson()).Duration.Ticks);
    }
}
