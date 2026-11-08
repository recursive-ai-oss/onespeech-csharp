using System.Buffers.Binary;
using System.Text.Json;
using OneSpeech.Audio;
using Xunit;

namespace OneSpeech.Tests;

public sealed class Qwen3FeatureTests
{
    [Fact]
    public void VariableLengthFeaturesMatchPublishedPreprocessor()
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_QWEN_PROBES");
        Assert.SkipWhen(string.IsNullOrEmpty(directory), "Set ONESPEECH_QWEN_PROBES to the black-box probe output directory.");
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory!, "probe.json")));
        float maximum = 0;
        foreach (var item in fixture.RootElement.GetProperty("features").EnumerateArray())
        {
            string name = item.GetProperty("name").GetString()!;
            float[] audio = ReadFloats(Path.Combine(directory!, name + ".audio.f32"));
            float[] reference = ReadFloats(Path.Combine(directory!, name + ".features.f32"));
            float[] actual = new WhisperLogMelSpectrogram(128).Extract(new AudioBuffer(audio, 16000), false, TestContext.Current.CancellationToken);
            Assert.Equal(reference.Length, actual.Length);
            Assert.Equal(128 * (audio.Length / 160), actual.Length);
            for (int i = 0; i < actual.Length; i++) maximum = MathF.Max(maximum, Math.Abs(reference[i] - actual[i]));
        }
        TestContext.Current.TestOutputHelper!.WriteLine("Maximum absolute feature error: " + maximum.ToString("G9", System.Globalization.CultureInfo.InvariantCulture));
        Assert.InRange(maximum, 0, 3e-5f);
    }

    private static float[] ReadFloats(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal(0, bytes.Length % 4);
        var values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return values;
    }
}
