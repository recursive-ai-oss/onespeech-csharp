using OneSpeech.Pipeline;
using Xunit;

namespace OneSpeech.Tests;

public sealed class DiarizationDefaultsTests
{
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
