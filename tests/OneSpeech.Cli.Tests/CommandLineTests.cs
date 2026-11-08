using Xunit;

namespace OneSpeech.Cli.Tests;

public sealed class CommandLineTests
{
    [Theory]
    [InlineData("--help")]
    [InlineData("transcribe --help")]
    [InlineData("identify --help")]
    public async Task HelpExplainsEphemeralEnrollmentAndExitCodes(string arguments)
    {
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(0, await CommandLine.RunAsync(arguments.Split(' '), output, error));
        Assert.Contains("never persisted", output.ToString());
        Assert.Contains("130 cancellation", output.ToString());
        Assert.Empty(error.ToString());
    }

    [Theory]
    [InlineData("unknown")]
    [InlineData("transcribe")]
    [InlineData("transcribe file --model")]
    [InlineData("transcribe file --format bogus")]
    [InlineData("transcribe file --model a --model b")]
    [InlineData("transcribe file --threshold 0.3")]
    [InlineData("identify file name enrollment --threshold NaN")]
    [InlineData("identify file name enrollment --threshold 1.01")]
    [InlineData("verify a b --format json")]
    [InlineData("detect-language file --language en")]
    [InlineData("models pull id --model ignored")]
    [InlineData("align audio text --unknown value")]
    public async Task UsageErrorsDoNotStartInference(string arguments)
    {
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(2, await CommandLine.RunAsync(arguments.Split(' '), output, error));
        Assert.Empty(output.ToString());
        Assert.Contains("--help", error.ToString());
    }

    [Fact]
    public async Task MissingAudioIsAnIoErrorNotUsage()
    {
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(1, await CommandLine.RunAsync(["diarize", Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))], output, error));
        Assert.Contains("FileNotFoundException:", error.ToString());
    }

    [Fact]
    public async Task OptionTerminatorPreservesLiteralHelpFilename()
    {
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(1, await CommandLine.RunAsync(["diarize", "--", "--help"], output, error));
        Assert.Contains("FileNotFoundException:", error.ToString());
        Assert.Empty(output.ToString());
    }

    [Fact]
    public async Task ModelListAndVersionWorkWithoutModels()
    {
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(0, await CommandLine.RunAsync(["models", "list"], output, error));
        Assert.Contains("openai/whisper-tiny", output.ToString());
        Assert.Equal(0, await CommandLine.RunAsync(["--version"], output, error));
        Assert.Empty(error.ToString());
    }
}
