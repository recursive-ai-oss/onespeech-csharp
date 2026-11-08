using System.Globalization;
using System.Text;

namespace OneSpeech.Evaluation;

/// <summary>A speaker segment, in seconds, on a recording/channel clock.</summary>
public sealed record DiarizationSegment(string Recording, string Channel, string Speaker, double Start, double End);

/// <summary>Bounded UTF-8 reader for the SPEAKER subset of RTTM (DIHARD III plan, Appendix C).</summary>
public static class Rttm
{
    /// <summary>Reads up to one million segments, 16,384 UTF-16 code units per line. Rejects invalid UTF-8, unsupported record types,
    /// nonfinite/negative times, nonpositive durations, and malformed fields with InvalidDataException. Leaves input open.</summary>
    public static IReadOnlyList<DiarizationSegment> Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var reader = new StreamReader(stream, new UTF8Encoding(false, true), false, 4096, true);
        var result = new List<DiarizationSegment>();
        var line = new StringBuilder();
        try
        {
            int c;
            while ((c = reader.Read()) != -1)
            {
                if (c == '\n') { Parse(); line.Clear(); }
                else { if (line.Length >= 16384) throw new InvalidDataException("RTTM line exceeds 16,384 characters."); line.Append((char)c); }
            }
            if (line.Length != 0) Parse();
        }
        catch (DecoderFallbackException ex) { throw new InvalidDataException("RTTM is not valid UTF-8.", ex); }
        return result;

        void Parse()
        {
            string text = line.ToString().Trim();
            if (text.Length == 0 || text.StartsWith('#')) return;
            var f = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (f.Length != 10 || f[0] != "SPEAKER" || f[1] == "<NA>" || f[7] == "<NA>" ||
                !int.TryParse(f[2], NumberStyles.None, CultureInfo.InvariantCulture, out int channel) || channel <= 0 ||
                !double.TryParse(f[3], NumberStyles.Float, CultureInfo.InvariantCulture, out double start) ||
                !double.TryParse(f[4], NumberStyles.Float, CultureInfo.InvariantCulture, out double duration) ||
                !double.IsFinite(start) || !double.IsFinite(duration) || start < 0 || duration <= 0 ||
                !double.IsFinite(start + duration) || start + duration <= start)
                throw new InvalidDataException("Expected SPEAKER, recording, positive channel, nonnegative onset, positive duration, and speaker ID in ten RTTM fields.");
            if (result.Count >= 1_000_000) throw new InvalidDataException("RTTM exceeds one million segments.");
            result.Add(new DiarizationSegment(f[1], channel.ToString(CultureInfo.InvariantCulture), f[7], start, start + duration));
        }
    }

    /// <summary>Converts assigned transcript turns to channel 1 of a supplied recording ID. Unassigned turns are omitted.</summary>
    public static IReadOnlyList<DiarizationSegment> FromTranscript(Transcript transcript, string recording)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        ArgumentException.ThrowIfNullOrWhiteSpace(recording);
        return transcript.Turns.Where(t => t.Speaker.Id != SpeakerLabel.Unassigned.Id)
            .Select(t => new DiarizationSegment(recording, "1", t.Speaker.Id, t.Start.TotalSeconds, t.End.TotalSeconds)).ToArray();
    }
}

/// <summary>Speaker-time counts and speaker-macro Jaccard error; rates are null when there is no reference speech.</summary>
public sealed record DiarizationScore(double ReferenceSeconds, double MissSeconds, double FalseAlarmSeconds, double ConfusionSeconds, double JaccardErrorSum, int ReferenceSpeakers)
{
    /// <summary>No collar; all overlaps scored. Can exceed one.</summary>
    public double? Der => ReferenceSeconds == 0 ? null : (MissSeconds + FalseAlarmSeconds + ConfusionSeconds) / ReferenceSeconds;
    /// <summary>Mean reference-speaker Jaccard error in [0,1], or null for no reference speakers.</summary>
    public double? Jer => ReferenceSpeakers == 0 ? null : JaccardErrorSum / ReferenceSpeakers;
}

/// <summary>No-collar, overlap-inclusive scoring with independent optimal one-to-one DER and JER speaker mappings.</summary>
public static class DiarizationMetrics
{
    /// <summary>Scores the union of recordings/channels and all times appearing in either side. Missing files are empty, not discarded.
    /// Same-speaker overlapping segments are unioned. At most 512 speakers per recording/channel are accepted.
    /// Throws ArgumentException for invalid segments; cancellation is checked per interval and assignment row.</summary>
    public static DiarizationScore Score(IEnumerable<DiarizationSegment> reference, IEnumerable<DiarizationSegment> hypothesis, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reference); ArgumentNullException.ThrowIfNull(hypothesis);
        var r = reference.ToArray(); var h = hypothesis.ToArray();
        foreach (var s in r.Concat(h))
            if (s is null || string.IsNullOrWhiteSpace(s.Recording) || string.IsNullOrWhiteSpace(s.Channel) || string.IsNullOrWhiteSpace(s.Speaker) ||
                !double.IsFinite(s.Start) || !double.IsFinite(s.End) || s.Start < 0 || s.End <= s.Start)
                throw new ArgumentException("Segments need IDs and finite nonnegative positive-duration intervals.");
        double total = 0, miss = 0, fa = 0, confusion = 0, jer = 0; int speakers = 0;
        foreach (var key in r.Concat(h).Select(s => (s.Recording, s.Channel)).Distinct().OrderBy(k => k.Recording, StringComparer.Ordinal).ThenBy(k => k.Channel, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var rr = r.Where(s => (s.Recording, s.Channel) == key).ToArray();
            var hh = h.Where(s => (s.Recording, s.Channel) == key).ToArray();
            var rn = rr.Select(s => s.Speaker).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
            var hn = hh.Select(s => s.Speaker).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
            if (rn.Length > 512 || hn.Length > 512) throw new ArgumentException("At most 512 speakers per recording/channel are supported.");
            int size = Math.Max(rn.Length, hn.Length);
            var intersections = new double[size, size];
            var rd = new double[size]; var hd = new double[size];
            var events = rr.SelectMany(s => new[]
                {
                    (Time: s.Start, Reference: true, Start: true, Speaker: Array.BinarySearch(rn, s.Speaker, StringComparer.Ordinal)),
                    (Time: s.End, Reference: true, Start: false, Speaker: Array.BinarySearch(rn, s.Speaker, StringComparer.Ordinal))
                }).Concat(hh.SelectMany(s => new[]
                {
                    (Time: s.Start, Reference: false, Start: true, Speaker: Array.BinarySearch(hn, s.Speaker, StringComparer.Ordinal)),
                    (Time: s.End, Reference: false, Start: false, Speaker: Array.BinarySearch(hn, s.Speaker, StringComparer.Ordinal))
                })).OrderBy(e => e.Time).ToArray();
            var ra = new Dictionary<int, int>(); var ha = new Dictionary<int, int>();
            double potentialCorrect = 0;
            int position = 0;
            while (position < events.Length)
            {
                cancellationToken.ThrowIfCancellationRequested();
                double time = events[position].Time;
                while (position < events.Length && events[position].Time == time)
                {
                    var e = events[position++];
                    var active = e.Reference ? ra : ha;
                    active.TryGetValue(e.Speaker, out int count);
                    if (e.Start) active[e.Speaker] = count + 1;
                    else if (count == 1) active.Remove(e.Speaker);
                    else active[e.Speaker] = count - 1;
                }
                if (position == events.Length) break;
                double duration = events[position].Time - time;
                total += duration * ra.Count;
                miss += duration * Math.Max(0, ra.Count - ha.Count);
                fa += duration * Math.Max(0, ha.Count - ra.Count);
                potentialCorrect += duration * Math.Min(ra.Count, ha.Count);
                foreach (int a in ra.Keys) { rd[a] += duration; foreach (int b in ha.Keys) intersections[a, b] += duration; }
                foreach (int b in ha.Keys) hd[b] += duration;
            }
            double matched = MaximumAssignment(intersections, cancellationToken);
            confusion += Math.Max(0, potentialCorrect - matched);
            var iou = new double[size, size];
            for (int a = 0; a < rn.Length; a++)
                for (int b = 0; b < hn.Length; b++)
                    iou[a, b] = intersections[a, b] / (rd[a] + hd[b] - intersections[a, b]);
            jer += rn.Length - MaximumAssignment(iou, cancellationToken);
            speakers += rn.Length;
        }
        if (!double.IsFinite(total) || !double.IsFinite(miss) || !double.IsFinite(fa) || !double.IsFinite(confusion) || !double.IsFinite(jer))
            throw new ArgumentException("Aggregate speaker-time exceeds the finite scoring range.");
        return new DiarizationScore(total, miss, fa, confusion, jer, speakers);
    }

    // Hungarian shortest augmenting paths, minimization of negative weights. Dummy rows/columns have zero weight.
    private static double MaximumAssignment(double[,] weights, CancellationToken cancellationToken)
    {
        int n = weights.GetLength(0);
        var u = new double[n + 1]; var v = new double[n + 1]; var p = new int[n + 1]; var way = new int[n + 1];
        for (int i = 1; i <= n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            p[0] = i; int j0 = 0;
            var min = Enumerable.Repeat(double.PositiveInfinity, n + 1).ToArray(); var used = new bool[n + 1];
            do
            {
                used[j0] = true; int i0 = p[j0], j1 = 0; double delta = double.PositiveInfinity;
                for (int j = 1; j <= n; j++) if (!used[j])
                {
                    double current = -weights[i0 - 1, j - 1] - u[i0] - v[j];
                    if (current < min[j]) { min[j] = current; way[j] = j0; }
                    if (min[j] < delta) { delta = min[j]; j1 = j; }
                }
                for (int j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else min[j] -= delta;
                j0 = j1;
            } while (p[j0] != 0);
            do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0 != 0);
        }
        double sum = 0;
        for (int j = 1; j <= n; j++) sum += weights[p[j] - 1, j - 1];
        return sum;
    }
}
