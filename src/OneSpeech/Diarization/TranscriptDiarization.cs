using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>A diarizer turn with relative activity used only to resolve concurrent speakers.</summary>
public sealed record ActiveTurn
{
    /// <summary>Turn on the original audio clock; requires nonnegative start and positive duration.</summary>
    public required Turn Turn { get; init; }
    /// <summary>Finite relative activity in [0,1]; defaults to 1 when unavailable. Not a calibrated probability.</summary>
    public double Activity { get; init; } = 1;
}

/// <summary>Shared deterministic word assignment and exclusive caption views. Never estimates missing times.</summary>
public static class TranscriptDiarization
{
    /// <summary>Assigns words by covered duration, resolving concurrent speech by activity. Ties use ordinal speaker ID.
    /// Replaces Turns and updates segment words without changing text, timing, or models. Null/zero-duration words become Unassigned.
    /// Each root word is attached to the same-speaker turn with greatest intersection (ties: earliest start).
    /// Throws ArgumentException for invalid turns and OperationCanceledException on cancellation.</summary>
    public static Transcript Merge(Transcript transcript, IEnumerable<ActiveTurn> turns, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        var input = Validate(turns);
        var intervals = Partition(input, cancellationToken);
        Word Assign(Word word)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var speaker = SpeakerLabel.Unassigned;
            bool overlap = false;
            if (word.Start is {} start && word.End is {} end && end > start)
            {
                var covered = new Dictionary<string, (SpeakerLabel Label, long Ticks)>(StringComparer.Ordinal);
                int first = FirstEndingAfter(intervals, start);
                for (int i = first; i < intervals.Count && intervals[i].Start < end; i++)
                {
                    var interval = intervals[i];
                    long ticks = Math.Min(end.Ticks, interval.End.Ticks) - Math.Max(start.Ticks, interval.Start.Ticks);
                    if (ticks <= 0) continue;
                    covered.TryGetValue(interval.Speaker.Id, out var old);
                    covered[interval.Speaker.Id] = (interval.Speaker, old.Ticks + ticks);
                    overlap |= interval.Overlap;
                }
                if (covered.Count != 0) speaker = covered.OrderByDescending(p => p.Value.Ticks).ThenBy(p => p.Key, StringComparer.Ordinal).First().Value.Label;
            }
            return word with { Speaker = speaker, Overlap = overlap };
        }
        var words = transcript.Words.Select(Assign).ToImmutableArray();
        var bySpeaker = input.Select((t, i) => (t.Turn, Index: i)).GroupBy(t => t.Turn.Speaker.Id).ToDictionary(g => g.Key, g => g.ToArray(), StringComparer.Ordinal);
        var turnWords = new List<Word>?[input.Length];
        foreach (var word in words)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (word.Speaker is null || !bySpeaker.TryGetValue(word.Speaker.Id, out var candidates) || word.Start is not {} start || word.End is not {} end) continue;
            int best = -1; long maximum = 0;
            foreach (var candidate in candidates)
            {
                if (candidate.Turn.Start >= end) break;
                long length = Math.Min(end.Ticks, candidate.Turn.End.Ticks) - Math.Max(start.Ticks, candidate.Turn.Start.Ticks);
                if (length > maximum) { maximum = length; best = candidate.Index; }
            }
            if (best >= 0) (turnWords[best] ??= []).Add(word);
        }
        return transcript with
        {
            Words = words,
            Segments = [.. transcript.Segments.Select(s => s with { Words = [.. s.Words.Select(Assign)] })],
            Turns = [.. input.Select((t, i) => t.Turn with
            {
                Words = turnWords[i] is {} assigned ? [.. assigned] : [],
                Text = turnWords[i] is {} textWords ? TextSpacing.Join(textWords.Select(w => w.Text)) : ""
            })]
        };
    }

    /// <summary>Splits overlapping turns into disjoint positive-duration intervals. Highest activity wins; ordinal ID breaks ties.
    /// Adjacent intervals with the same speaker are coalesced. Text and words are empty: no transcript is inferred.
    /// Throws ArgumentException for invalid turns; checks cancellation at each boundary.</summary>
    public static ImmutableArray<Turn> Exclusive(IEnumerable<ActiveTurn> turns, CancellationToken cancellationToken = default)
    {
        var intervals = Partition(Validate(turns), cancellationToken);
        var result = ImmutableArray.CreateBuilder<Turn>();
        foreach (var interval in intervals)
        {
            if (result.Count != 0 && result[^1].Speaker.Id == interval.Speaker.Id && result[^1].End == interval.Start)
                result[^1] = result[^1] with { End = interval.End };
            else result.Add(new Turn { Speaker = interval.Speaker, Start = interval.Start, End = interval.End, Text = "" });
        }
        return result.ToImmutable();
    }

    private sealed record Interval(TimeSpan Start, TimeSpan End, SpeakerLabel Speaker, bool Overlap);

    private static int FirstEndingAfter(List<Interval> intervals, TimeSpan start)
    {
        int low = 0, high = intervals.Count;
        while (low < high) { int mid = low + (high - low) / 2; if (intervals[mid].End <= start) low = mid + 1; else high = mid; }
        return low;
    }

    private static List<Interval> Partition(ActiveTurn[] input, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var events = input.SelectMany((t, i) => new[] { (Time: t.Turn.Start, Start: true, Index: i), (Time: t.Turn.End, Start: false, Index: i) }).OrderBy(e => e.Time).ToArray();
        var active = new SortedSet<int>(Comparer<int>.Create((a, b) =>
        {
            int order = input[b].Activity.CompareTo(input[a].Activity);
            if (order == 0) order = StringComparer.Ordinal.Compare(input[a].Turn.Speaker.Id, input[b].Turn.Speaker.Id);
            return order == 0 ? a.CompareTo(b) : order;
        }));
        var speakers = new Dictionary<string, int>(StringComparer.Ordinal);
        var result = new List<Interval>();
        int position = 0;
        while (position < events.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var time = events[position].Time;
            while (position < events.Length && events[position].Time == time)
            {
                var e = events[position++];
                string id = input[e.Index].Turn.Speaker.Id;
                speakers.TryGetValue(id, out int count);
                if (e.Start) { active.Add(e.Index); speakers[id] = count + 1; }
                else { active.Remove(e.Index); if (count == 1) speakers.Remove(id); else speakers[id] = count - 1; }
            }
            if (position < events.Length && active.Count != 0)
                result.Add(new Interval(time, events[position].Time, input[active.Min].Turn.Speaker, speakers.Count > 1));
        }
        return result;
    }

    private static ActiveTurn[] Validate(IEnumerable<ActiveTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(turns);
        var input = turns.ToArray();
        foreach (var item in input)
            if (item is null || item.Turn is null || item.Turn.Start < TimeSpan.Zero || item.Turn.End <= item.Turn.Start ||
                item.Turn.Speaker is null || string.IsNullOrWhiteSpace(item.Turn.Speaker.Id) || item.Turn.Speaker.Id == SpeakerLabel.Unassigned.Id ||
                !double.IsFinite(item.Activity) || item.Activity < 0 || item.Activity > 1)
                throw new ArgumentException("Turns require a speaker, nonnegative start, positive duration and finite activity in [0,1].", nameof(turns));
        return input.OrderBy(t => t.Turn.Start).ThenBy(t => t.Turn.Speaker.Id, StringComparer.Ordinal).ToArray();
    }
}
