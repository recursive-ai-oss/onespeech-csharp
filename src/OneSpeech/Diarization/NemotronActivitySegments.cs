using System.Collections.Immutable;

namespace OneSpeech.Diarization;

// Retains change points, not a recording-long probability matrix. Relative activity preserves the winning
// channel of every 10 ms frame; lower-ranked simultaneous channels remain present for overlap scoring.
internal sealed class NemotronActivitySegments
{
    private readonly List<ActiveTurn> turns = [];
    private int mask, winner = -1;
    private long start, end;
    internal void Append(ReadOnlySpan<float> probabilities, int frames, long firstMelFrame, long sampleEnd)
    {
        if (frames < 0 || probabilities.Length < frames * 8) throw new ArgumentException("Probability shape does not match frames.");
        for (int frame = 0; frame < frames; frame++)
        {
            int nextMask = 0, nextWinner = -1;
            float best = float.NegativeInfinity;
            for (int speaker = 0; speaker < 8; speaker++)
            {
                float value = probabilities[frame * 8 + speaker];
                if (!float.IsFinite(value) || value < 0 || value > 1) throw new InvalidDataException("Invalid speaker probability.");
                if (value < .5f) continue;
                nextMask |= 1 << speaker;
                if (value > best) { best = value; nextWinner = speaker; }
            }
            long left = checked((firstMelFrame + frame) * TimeSpan.TicksPerMillisecond * 10);
            long right = Math.Min(left + TimeSpan.TicksPerMillisecond * 10, checked(sampleEnd * (TimeSpan.TicksPerSecond / 16000)));
            if (right <= left) continue;
            if (nextMask != mask || nextWinner != winner) { Flush(); start = left; mask = nextMask; winner = nextWinner; }
            end = right;
        }
    }
    internal ImmutableArray<ActiveTurn> Complete()
    {
        Flush(); mask = 0; winner = -1;
        return turns.OrderBy(t => t.Turn.Start).ThenBy(t => t.Turn.Speaker.Id, StringComparer.Ordinal).ToImmutableArray();
    }
    private void Flush()
    {
        if (end <= start) return;
        for (int speaker = 0; speaker < 8; speaker++)
            if ((mask & (1 << speaker)) != 0)
                turns.Add(new ActiveTurn { Turn = new Turn { Start = TimeSpan.FromTicks(start), End = TimeSpan.FromTicks(end), Speaker = new SpeakerLabel { Id = "speaker_" + speaker }, Text = "" }, Activity = speaker == winner ? 1 : .5 });
    }
    internal static ImmutableArray<Turn> Coalesce(ImmutableArray<ActiveTurn> activity)
    {
        List<Turn> result = [];
        var last = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var item in activity)
        {
            var turn = item.Turn;
            if (last.TryGetValue(turn.Speaker.Id, out int index) && result[index].End == turn.Start)
                result[index] = result[index] with { End = turn.End };
            else { last[turn.Speaker.Id] = result.Count; result.Add(turn); }
        }
        return result.OrderBy(t => t.Start).ThenBy(t => t.Speaker.Id, StringComparer.Ordinal).ToImmutableArray();
    }
}
