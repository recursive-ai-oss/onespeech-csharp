using System.Text.Json;
using OneSpeech.Audio;

namespace OneSpeech.Evaluation;

// Development-only physical microphone layout calibration. No new speech waveform is synthesized.
internal static class ChannelCalibration
{
    private static readonly string[] Meetings = ["ES2011a", "ES2011b"];
    private static readonly string[] Hashes = ["4950ebec42f08e3caaf8d038d9da9ceeda89ececb6dbc7f0f3bd25802e919fe9", "b3ccaeb285790058bb9ed598d56f61ff287c35ea4285d18701df227e20358fd1", "7a4e9a0a5ca43a37d27879e86cdbd1f51c94f965df601b420092df79156073af", "da0ee28e72d18cf51be655a16bc1dbd632543ff46ef15903f8ba83f6bd5aceb3"];
    private static readonly double[] Decibels = [10, 15, 20, 25, 30];
    private static readonly double[] Correlations = [0.2, 0.3, 0.4, 0.5, 0.6];
    private static readonly double[] Leaks = [0, 0.01, 0.25, 0.5];
    internal static void Run(string root, string output)
    {
        var cases = new List<ChannelCase>();
        for (int i = 0; i < Meetings.Length; i++)
        {
            string left = Path.Combine(root, "downloads", Meetings[i] + ".Headset-0.wav"), right = Path.Combine(root, "downloads", Meetings[i] + ".Headset-1.wav");
            if (DiarizationRun.HashFile(left) != Hashes[2 * i] || DiarizationRun.HashFile(right) != Hashes[2 * i + 1]) throw new InvalidDataException("Development headset SHA-256 mismatch.");
            var frames = Features(left, right);
            foreach (double leak in Leaks) cases.Add(new ChannelCase(Meetings[i], leak, false, leak < 0.1, frames));
            cases.Add(new ChannelCase(Meetings[i], 0, true, false, frames));
        }
        var ranking = new List<Choice>();
        foreach (double db in Decibels) foreach (double correlation in Correlations)
        {
            int tp = 0, tn = 0, fp = 0, fn = 0;
            foreach (var item in cases)
            {
                bool predicted = Measure(item, db, correlation).Separated;
                if (item.Separated && predicted) tp++; else if (item.Separated) fn++; else if (predicted) fp++; else tn++;
            }
            double error = 0.5 * ((double)fn / (tp + fn) + (double)fp / (tn + fp));
            ranking.Add(new Choice(db, correlation, error, tp, tn, fp, fn));
        }
        var ordered = ranking.OrderBy(c => c.BalancedError).ThenBy(c => Math.Abs(c.Decibels - 20) / 10 + Math.Abs(c.Correlation - 0.3)).ThenByDescending(c => c.Decibels).ThenBy(c => c.Correlation).ToArray();
        var best = ordered[0]; Directory.CreateDirectory(output);
        using var file = File.Create(Path.Combine(output, "channel-selection.json")); using var w = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        w.WriteStartObject(); w.WriteString("policy", "AMI-dev ES2011a/b headset0/1 layout probes; minimize balanced error, then smallest change from 20dB/0.3, then higher dB/lower correlation. Fixed 20ms frames, 3 dominant frames/channel, floor1e-8.");
        w.WriteString("limitations", "Ten correlated layout/leakage proxies from two meetings, not an independent telephony accuracy estimate. Leakage is calculated on second-order frame statistics only; no speech waveform or embedding saved.");
        w.WriteNumber("dominanceDecibels", best.Decibels); w.WriteNumber("maximumNormalizedDot", best.Correlation); w.WriteNumber("balancedError", best.BalancedError);
        w.WriteStartArray("grid"); foreach (var choice in ordered)
        { w.WriteStartObject(); w.WriteNumber("decibels", choice.Decibels); w.WriteNumber("correlation", choice.Correlation); w.WriteNumber("balancedError", choice.BalancedError); w.WriteNumber("tp", choice.Tp); w.WriteNumber("tn", choice.Tn); w.WriteNumber("fp", choice.Fp); w.WriteNumber("fn", choice.Fn); w.WriteEndObject(); } w.WriteEndArray();
        w.WriteStartArray("cases"); foreach (var item in cases)
        {
            var measured = Measure(item, best.Decibels, best.Correlation); int i = Array.IndexOf(Meetings, item.Id);
            w.WriteStartObject(); w.WriteString("id", item.Id); w.WriteString("headset0Sha256", Hashes[2 * i]); w.WriteString("headset1Sha256", Hashes[2 * i + 1]); w.WriteNumber("leakage", item.Leak); w.WriteBoolean("dualMono", item.DualMono);
            w.WriteBoolean("expectedSeparated", item.Separated); w.WriteBoolean("predictedSeparated", measured.Separated); w.WriteNumber("normalizedDot", measured.Correlation); w.WriteNumber("leftDominantFrames", measured.Left); w.WriteNumber("rightDominantFrames", measured.Right); w.WriteEndObject();
        } w.WriteEndArray(); w.WriteEndObject();
        Console.WriteLine($"Channel selection: {best.Decibels} dB, normalized dot <= {best.Correlation}; balanced error {best.BalancedError}; TP={best.Tp} TN={best.Tn} FP={best.Fp} FN={best.Fn}");
    }
    private static List<Frame> Features(string left, string right)
    {
        using var a = AudioFileSource.Open(left, CancellationToken.None); using var b = AudioFileSource.Open(right, CancellationToken.None);
        if (a.Channels != 1 || b.Channels != 1) throw new InvalidDataException("Headset channels must be mono.");
        var x = new float[320]; var y = new float[320]; var result = new List<Frame>();
        int Fill(AudioFileSource source, float[] buffer) { int used = 0, count; while (used < buffer.Length && (count = source.Read(buffer.AsSpan(used), CancellationToken.None)) != 0) used += count; return used; }
        while (true)
        {
            int count = Fill(a, x), other = Fill(b, y); if (count != other) throw new InvalidDataException("Headset durations differ."); if (count == 0) break;
            double xx = 0, yy = 0, xy = 0;
            for (int i = 0; i < count; i++) { xx += (double)x[i] * x[i]; yy += (double)y[i] * y[i]; xy += (double)x[i] * y[i]; }
            result.Add(new Frame(xx, yy, xy));
        }
        return result;
    }
    private static Measurement Measure(ChannelCase item, double decibels, double correlation)
    {
        double aa = 0, bb = 0, ab = 0, ratio = Math.Pow(10, decibels / 10), alpha = item.Leak; int left = 0, right = 0;
        foreach (var f in item.Frames)
        {
            double a, b, cross;
            if (item.DualMono) a = b = cross = (f.Xx + f.Yy + 2 * f.Xy) / 4;
            else { double scale = (1 + alpha) * (1 + alpha); a = (f.Xx + 2 * alpha * f.Xy + alpha * alpha * f.Yy) / scale; b = (f.Yy + 2 * alpha * f.Xy + alpha * alpha * f.Xx) / scale; cross = (alpha * (f.Xx + f.Yy) + (1 + alpha * alpha) * f.Xy) / scale; }
            aa += a; bb += b; ab += cross;
            if (a > 1e-8 * 320 && a >= ratio * b) left++; if (b > 1e-8 * 320 && b >= ratio * a) right++;
        }
        double dot = aa > 0 && bb > 0 ? Math.Abs(ab / Math.Sqrt(aa * bb)) : 1;
        return new Measurement(left >= 3 && right >= 3 && dot <= correlation, dot, left, right);
    }
    private sealed record Frame(double Xx, double Yy, double Xy);
    private sealed record ChannelCase(string Id, double Leak, bool DualMono, bool Separated, List<Frame> Frames);
    private sealed record Choice(double Decibels, double Correlation, double BalancedError, int Tp, int Tn, int Fp, int Fn);
    private sealed record Measurement(bool Separated, double Correlation, int Left, int Right);
}
