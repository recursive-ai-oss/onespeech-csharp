using System.Security.Cryptography;
using System.Text;
using OneSpeech.Audio;

internal static class Fixtures
{
    internal static string Hash(string file)
    {
        using var stream = File.OpenRead(file);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
    internal static double Duration(string file)
    {
        using var source = AudioFileSource.Open(file, default);
        var buffer = new float[16000 * source.Channels];
        long samples = 0; int n;
        while ((n = source.Read(buffer, default)) != 0) samples += n;
        return samples / 16000d;
    }
    internal static float[] ReadMono(string file)
    {
        using var source = AudioFileSource.Open(file, default);
        var data = new List<float>();
        var buffer = new float[16000 * source.Channels]; int n;
        while ((n = source.Read(buffer, default)) != 0)
            for (int i = 0; i < n * source.Channels; i += source.Channels)
            {
                float sum = 0; for (int c = 0; c < source.Channels; c++) sum += buffer[i + c];
                data.Add(sum / source.Channels);
            }
        return data.ToArray();
    }
    internal static void Prepare(string corpus, string meeting, string directory)
    {
        Directory.CreateDirectory(directory);
        string[] files = Directory.GetFiles(corpus, "*.flac", SearchOption.AllDirectories).Order(StringComparer.Ordinal).ToArray();
        if (files.Length == 0) throw new ArgumentException("No LibriSpeech FLACs found.");
        var sources = new HashSet<string>(StringComparer.Ordinal);
        Write("warm.wav", 10, files.Reverse().ToArray());
        string last = Path.GetFileNameWithoutExtension(files[^1]);
        string warmText = File.ReadLines(Directory.GetFiles(Path.GetDirectoryName(files[^1])!, "*.trans.txt").Single()).Single(l => l.StartsWith(last + " ", StringComparison.Ordinal));
        File.WriteAllText(Path.Combine(directory, "warm-align.txt"), warmText[(last.Length + 1)..]);
        Write("clip60.wav", 60, files);
        Write("clip600.wav", 600, files);
        Write("long.wav", 3600, files);
        Write("ami.wav", Duration(meeting), [meeting]);
        // Use transcript words for the 60 s prefix, including the final truncated utterance.
        var text = new StringBuilder(); double seconds = 0;
        foreach (string file in files)
        {
            string utterance = Path.GetFileNameWithoutExtension(file);
            string transcript = Directory.GetFiles(Path.GetDirectoryName(file)!, "*.trans.txt").Single();
            string line = File.ReadLines(transcript).Single(l => l.StartsWith(utterance + " ", StringComparison.Ordinal));
            text.Append(line.AsSpan(utterance.Length + 1)); text.Append(' ');
            seconds += Duration(file); if (seconds >= 60) break;
        }
        File.WriteAllText(Path.Combine(directory, "align.txt"), text.ToString().Trim());
        File.WriteAllLines(Path.Combine(directory, "sources.tsv"), sources.Order(StringComparer.Ordinal).Select(f => $"{f}\t{Hash(f)}\tCC-BY-4.0"));
        File.WriteAllText(Path.Combine(directory, "LICENSE-NOTE"), "LibriSpeech, OpenSLR 12, CC-BY-4.0; AMI Meeting Corpus, AMI Consortium, CC-BY-4.0. Derived mono 16 kHz PCM16 prefixes; long input concatenates sorted LibriSpeech utterances without inserted silence. Evaluation only. Retain upstream sources.tsv and licence notes.\n");
        File.WriteAllLines(Path.Combine(directory, "SHA256SUMS"), Directory.GetFiles(directory, "*.wav").Order(StringComparer.Ordinal).Select(f => Hash(f) + "  " + Path.GetFileName(f)));

        void Write(string name, double duration, string[] inputs)
        {
            int remaining = checked((int)Math.Round(duration * 16000));
            using var stream = File.Create(Path.Combine(directory, name));
            using var writer = new BinaryWriter(stream, Encoding.UTF8);
            writer.Write("RIFF"u8); writer.Write(checked(36 + remaining * 2)); writer.Write("WAVEfmt "u8);
            writer.Write(16); writer.Write((short)1); writer.Write((short)1); writer.Write(16000); writer.Write(32000); writer.Write((short)2); writer.Write((short)16);
            writer.Write("data"u8); writer.Write(remaining * 2);
            while (remaining > 0)
                foreach (string file in inputs)
                {
                    sources.Add(file);
                    using var source = AudioFileSource.Open(file, default);
                    var buffer = new float[16000 * source.Channels]; int n;
                    while (remaining > 0 && (n = source.Read(buffer, default)) != 0)
                        for (int i = 0; i < n * source.Channels && remaining > 0; i += source.Channels)
                        {
                            float sum = 0; for (int c = 0; c < source.Channels; c++) sum += buffer[i + c];
                            writer.Write((short)Math.Clamp((int)MathF.Round(sum / source.Channels * 32768), short.MinValue, short.MaxValue)); remaining--;
                        }
                    if (remaining == 0) break;
                }
        }
    }
}
