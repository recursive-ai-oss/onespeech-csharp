using System.Buffers.Binary;
using System.Globalization;
using OneSpeech;
using OneSpeech.Audio;

if (args.Length is < 1 or > 2) throw new ArgumentException("Specify the NeMo oracle fixture directory.");
var extractor = new NemoLogMelSpectrogram();
foreach (string line in File.ReadLines(Path.Combine(args[0], "manifest.tsv")))
{
    string[] cells = line.Split('\t');
    float[] pcm = Read(Path.Combine(args[0], cells[0] + ".pcm"));
    float[] expected = Read(Path.Combine(args[0], cells[0] + ".mel"));
    float[] actual = extractor.Extract(new AudioBuffer(pcm, 16000), normalize: args.Length == 1);
    if (actual.Length != expected.Length) throw new InvalidDataException($"{cells[0]} shape mismatch {actual.Length} != {expected.Length}");
    if (int.Parse(cells[2], CultureInfo.InvariantCulture) != NemoLogMelSpectrogram.GetFrameCount(pcm.Length)) throw new InvalidDataException("Valid-frame count differs.");
    double sum = 0, max = 0;
    int worst = 0;
    for (int i = 0; i < actual.Length; i++)
    {
        double error = Math.Abs(actual[i] - expected[i]);
        if (error > max) { max = error; worst = i; }
        sum += error;
    }
    double tolerance = args.Length == 2 ? 1.1e-4 : cells[0] == "dc" ? 1e-3 : 1e-4;
    if (max > tolerance || !double.IsFinite(sum)) throw new InvalidDataException($"{cells[0]} error {max} exceeds {tolerance}.");
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{cells[0]}\t{NemoLogMelSpectrogram.GetFrameCount(pcm.Length)}\tmax={max:G9}\tmean={sum / actual.Length:G9}\tworst={worst}\tours={actual[worst]:G9}\tref={expected[worst]:G9}"));
}
static float[] Read(string path)
{
    byte[] bytes = File.ReadAllBytes(path);
    if (bytes.Length % 4 != 0) throw new InvalidDataException("Truncated float32 fixture.");
    float[] values = new float[bytes.Length / 4];
    for (int i = 0; i < values.Length; i++) values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
    return values;
}
