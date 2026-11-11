using System.Formats.Tar;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Xml;
using OneSpeech.Audio;

if (args.Length < 2 || args[0] is not ("ami" or "alimeeting"))
{
    Console.Error.WriteLine("Usage: OneSpeech.DiarizationData ami DATA_ROOT [MEETING ...] | alimeeting DATA_ROOT [all] (archives must be fetched and SHA-256 verified first)");
    return 2;
}
try
{
    string root = Path.GetFullPath(args[1]); string prepared = Path.Combine(root, "prepared"); Directory.CreateDirectory(prepared);
    var rows = new List<Prepared>();
    if (args[0] == "ami")
    {
        using var archive = ZipFile.OpenRead(Path.Combine(root, "downloads", "ami_public_manual_1.6.2.zip"));
        foreach (string id in args.Length > 2 ? args[2..] : new[] { "ES2011a", "ES2011b", "ES2004a", "ES2004b" })
        {
            var intervals = new List<Interval>();
            foreach (var entry in archive.Entries.Where(e => e.FullName.StartsWith("words/" + id + ".", StringComparison.Ordinal) && e.FullName.EndsWith(".words.xml", StringComparison.Ordinal)))
            {
                string speaker = Path.GetFileName(entry.FullName).Split('.')[1];
                using var input = entry.Open();
                using var reader = XmlReader.Create(input, new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, MaxCharactersInDocument = 16 * 1024 * 1024 });
                while (reader.Read()) if (reader.NodeType == XmlNodeType.Element && reader.LocalName == "w")
                {
                    var start = reader.GetAttribute("starttime"); var end = reader.GetAttribute("endtime");
                    if (start is null || end is null) continue;
                    double a = Number(start), b = Number(end);
                    if (b < a) throw new InvalidDataException("Reversed AMI word interval.");
                    if (b > a) intervals.Add(new Interval(speaker, a, b));
                }
            }
            if (intervals.Count == 0) throw new InvalidDataException("Missing AMI word annotations: " + id);
            string source = Path.Combine(root, "downloads", id + ".Mix-Headset.wav");
            rows.Add(Prepare(root, source, id, id.StartsWith("ES2011", StringComparison.Ordinal) ? "dev" : "test", "CC-BY-4.0", intervals));
        }
    }
    else
    {
        string raw = Path.Combine(root, "raw"); Directory.CreateDirectory(raw);
        using (var file = File.OpenRead(Path.Combine(root, "downloads", "Eval_Ali.tar.gz")))
        using (var gzip = new GZipStream(file, CompressionMode.Decompress))
        using (var tar = new TarReader(gzip))
        {
            TarEntry? entry;
            while ((entry = tar.GetNextEntry()) is not null)
            {
                // Only these data files are selected. No archive path or link is followed.
                string name = Path.GetFileName(entry.Name);
                bool selected = args.Length > 2 && args[2] == "all"
                    ? name.EndsWith(".TextGrid", StringComparison.Ordinal) || name.EndsWith(".wav", StringComparison.Ordinal)
                    : name is "R8001_M8004_MS801.wav" or "R8003_M8001_MS801.wav" or "R8001_M8004.TextGrid" or "R8003_M8001.TextGrid";
                if (!selected || !entry.Name.StartsWith("Eval_Ali/Eval_Ali_far/", StringComparison.Ordinal)) continue;
                if (entry.EntryType is not (TarEntryType.RegularFile or TarEntryType.V7RegularFile) || entry.DataStream is null) throw new InvalidDataException("Selected archive entry is not regular data.");
                string extracted = Path.Combine(raw, name);
                if (name.EndsWith(".wav", StringComparison.Ordinal))
                {
                    string temporary = extracted + ".source.wav";
                    using (var output = File.Create(temporary)) entry.DataStream.CopyTo(output);
                    MonoWave(temporary, extracted);
                    File.Delete(temporary);
                }
                else
                {
                    using var output = File.Create(extracted); entry.DataStream.CopyTo(output);
                }
            }
        }
        foreach (string id in args.Length > 2 && args[2] == "all"
            ? Directory.EnumerateFiles(raw, "*.wav").Select(Path.GetFileNameWithoutExtension).OfType<string>().Order(StringComparer.Ordinal).ToArray()
            : new[] { "R8001_M8004_MS801", "R8003_M8001_MS801" })
        {
            string annotation = string.Join('_', id.Split('_').Take(2)) + ".TextGrid";
            var intervals = TextGrid(Path.Combine(raw, annotation));
            if (intervals.Count == 0) throw new InvalidDataException("Missing AliMeeting interval annotations: " + id);
            rows.Add(Prepare(root, Path.Combine(raw, id + ".wav"), id, "eval", "CC-BY-SA-4.0", intervals));
        }
    }
    using (var file = File.Create(Path.Combine(root, "manifest.jsonl")))
    {
        foreach (var row in rows)
        {
            using var buffer = new MemoryStream();
            using (var json = new Utf8JsonWriter(buffer))
            {
                json.WriteStartObject(); json.WriteString("id", row.Id); json.WriteString("split", row.Split); json.WriteString("license", row.License);
                json.WriteString("audio", row.Audio); json.WriteString("reference", row.Reference); json.WriteString("audioSha256", row.AudioHash); json.WriteString("referenceSha256", row.ReferenceHash);
                json.WriteNumber("durationSeconds", row.Duration); json.WriteNumber("referenceSegments", row.Segments); json.WriteNumber("referenceSpeakers", row.Speakers);
                json.WriteString("preparation", "diarization-data/1; first input channel; 16kHz PCM16; timed words for AMI, nonempty intervals for AliMeeting; no gap bridging"); json.WriteEndObject();
            }
            buffer.Position = 0; buffer.CopyTo(file); file.WriteByte((byte)'\n');
        }
    }
    Console.WriteLine($"Prepared {rows.Count} recordings; source and derived data remain in the specified data root.");
    return 0;
}
catch (Exception ex) when (ex is IOException or InvalidOperationException or XmlException or ArgumentException or OneSpeech.OneSpeechException)
{ Console.Error.WriteLine(ex.Message); return 2; }

static double Number(string value)
{
    double number = double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
    if (!double.IsFinite(number) || number < 0 || number > TimeSpan.MaxValue.TotalSeconds) throw new InvalidDataException("Invalid annotation time.");
    return number;
}

static List<Interval> TextGrid(string path)
{
    var result = new List<Interval>(); string speaker = ""; bool interval = false; bool tier = false; double? start = null, end = null;
    using var reader = new StreamReader(path, new UTF8Encoding(false, true), true);
    string? line;
    while ((line = reader.ReadLine()) is not null)
    {
        string value = line.Trim();
        if (value.StartsWith("item [", StringComparison.Ordinal)) { tier = false; interval = false; speaker = ""; }
        else if (value.StartsWith("class =", StringComparison.Ordinal)) tier = value.EndsWith("\"IntervalTier\"", StringComparison.Ordinal);
        else if (value.StartsWith("name =", StringComparison.Ordinal)) speaker = Quoted(value);
        else if (value.StartsWith("intervals [", StringComparison.Ordinal)) { interval = true; start = end = null; }
        else if (interval && value.StartsWith("xmin =", StringComparison.Ordinal)) start = Number(value[(value.IndexOf('=') + 1)..].Trim());
        else if (interval && value.StartsWith("xmax =", StringComparison.Ordinal)) end = Number(value[(value.IndexOf('=') + 1)..].Trim());
        else if (interval && value.StartsWith("text =", StringComparison.Ordinal))
        {
            if (start is null || end is null || end < start || speaker.Length == 0 || speaker.Any(char.IsWhiteSpace)) throw new InvalidDataException("Malformed TextGrid interval.");
            if (tier && end > start && Quoted(value).Length != 0) result.Add(new Interval(speaker, start.Value, end.Value));
            interval = false;
        }
    }
    return result;
}

static string Quoted(string line)
{
    int first = line.IndexOf('"'), last = line.LastIndexOf('"');
    if (first < 0 || last <= first) throw new InvalidDataException("Expected quoted TextGrid value.");
    return line[(first + 1)..last].Replace("\"\"", "\"", StringComparison.Ordinal).Trim();
}

static Prepared Prepare(string root, string source, string id, string split, string license, List<Interval> intervals)
{
    string audioRelative = "prepared/" + id + ".wav", referenceRelative = "prepared/" + id + ".rttm";
    double duration = MonoWave(source, Path.Combine(root, audioRelative));
    var merged = new List<Interval>();
    foreach (var group in intervals.GroupBy(i => i.Speaker, StringComparer.Ordinal))
    {
        Interval? previous = null;
        foreach (var item in group.OrderBy(i => i.Start).ThenBy(i => i.End))
        {
            if (item.End > duration + 0.001) throw new InvalidDataException("Annotation exceeds audio duration: " + id);
            if (previous is not null && item.Start <= previous.End) previous = previous with { End = Math.Max(previous.End, item.End) };
            else { if (previous is not null) merged.Add(previous); previous = item; }
        }
        if (previous is not null) merged.Add(previous);
    }
    using (var output = new StreamWriter(Path.Combine(root, referenceRelative), false, new UTF8Encoding(false)))
        foreach (var item in merged.OrderBy(i => i.Start).ThenBy(i => i.Speaker, StringComparer.Ordinal))
            output.WriteLine(FormattableString.Invariant($"SPEAKER {id} 1 {item.Start:R} {item.End - item.Start:R} <NA> <NA> {item.Speaker} <NA> <NA>"));
    string Hash(string relative) { using var input = File.OpenRead(Path.Combine(root, relative)); return Convert.ToHexStringLower(SHA256.HashData(input)); }
    return new Prepared(id, split, license, audioRelative, referenceRelative, Hash(audioRelative), Hash(referenceRelative), duration, merged.Count, merged.Select(i => i.Speaker).Distinct().Count());
}

static double MonoWave(string input, string output)
{
    using var source = AudioFileSource.Open(input, CancellationToken.None);
    using var file = File.Create(output); using var writer = new BinaryWriter(file);
    writer.Write("RIFF"u8); writer.Write(0); writer.Write("WAVEfmt "u8); writer.Write(16); writer.Write((short)1); writer.Write((short)1);
    writer.Write(16000); writer.Write(32000); writer.Write((short)2); writer.Write((short)16); writer.Write("data"u8); writer.Write(0);
    var block = new float[4096 * source.Channels]; long frames = 0; int count;
    while ((count = source.Read(block, CancellationToken.None)) != 0)
    {
        frames += count;
        if (frames > (int.MaxValue - 36) / 2) throw new InvalidDataException("Prepared WAV exceeds RIFF size bound.");
        for (int i = 0; i < count; i++) writer.Write((short)Math.Clamp(Math.Round(block[i * source.Channels] * 32768.0), short.MinValue, short.MaxValue));
    }
    file.Position = 4; writer.Write(checked(36 + (int)frames * 2)); file.Position = 40; writer.Write(checked((int)frames * 2));
    return frames / 16000.0;
}

internal sealed record Interval(string Speaker, double Start, double End);
internal sealed record Prepared(string Id, string Split, string License, string Audio, string Reference, string AudioHash, string ReferenceHash, double Duration, int Segments, int Speakers);
