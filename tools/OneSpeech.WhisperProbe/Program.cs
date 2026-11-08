using System.Buffers.Binary;
using System.Globalization;
using OneSpeech;
using OneSpeech.Audio;

if (args.Length != 1) { Console.Error.WriteLine("Usage: OneSpeech.WhisperProbe <oracle-output-directory>"); return 2; }
Console.WriteLine("clip,bins,reference,max_abs,mean_abs");
foreach (string path in Directory.GetFiles(args[0], "*.pcm").Order(StringComparer.Ordinal))
{
    float[] audio = ReadFloats(path);
    foreach (int bins in new[] { 80, 128 })
    {
        float[] actual = new WhisperLogMelSpectrogram(bins).Extract(new AudioBuffer(audio, 16000));
        foreach (string reference in new[] { "whisper", "transformers" })
        {
            string expectedPath = Path.Combine(args[0], $"{Path.GetFileNameWithoutExtension(path)}.{bins}.{reference}.f32");
            float[] expected = ReadFloats(expectedPath);
            if (actual.Length != expected.Length) throw new InvalidDataException("Feature shape mismatch.");
            double max = 0, sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                double error = Math.Abs(actual[i] - expected[i]);
                if (!double.IsFinite(error)) throw new InvalidDataException("Nonfinite feature error.");
                max = Math.Max(max, error);
                sum += error;
            }
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{Path.GetFileNameWithoutExtension(path)},{bins},{reference},{max:G9},{sum / actual.Length:G9}"));
        }
    }
}
return 0;

static float[] ReadFloats(string path)
{
    using FileStream stream = File.OpenRead(path);
    if (stream.Length % 4 != 0 || stream.Length > 4 * 480000) throw new InvalidDataException("Invalid float file length.");
    byte[] bytes = new byte[stream.Length];
    stream.ReadExactly(bytes);
    float[] result = new float[bytes.Length / 4];
    for (int i = 0; i < result.Length; i++) result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
    return result;
}
