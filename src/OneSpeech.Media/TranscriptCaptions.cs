using System.Text;
using OneSpeech.Diarization;
using RecursiveAI.OneMedia;
using RecursiveAI.OneMedia.Captions;

namespace OneSpeech.Media;

/// <summary>Caption conversion over OneMedia's cue model. Times remain on the input audio clock.</summary>
public static class TranscriptCaptions
{
    /// <summary>Builds captions from segments (root words when segments are absent). Splits at speaker changes, 84 UTF-16 characters or 6 seconds and wraps at 42 characters without splitting a word. A single long word is not truncated. A segment with any untimed word remains one wrapped cue (no fabricated split boundaries). Turn assignment uses core's majority-overlap rules. Exclusive display clips overlapping cue ends to the next cue start, dropping fully covered cues; original absolute word times remain unchanged. Untimed words retain null span times; segment bounds may time their cue, otherwise its times are Unknown. Cancellation is checked per word. Does not estimate times.</summary>
    public static SubtitleDocument ToSubtitles(this Transcript transcript, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        cancellationToken.ThrowIfCancellationRequested();
        if (!transcript.Turns.IsEmpty)
            transcript = TranscriptDiarization.Merge(transcript, transcript.Turns.Select(t => new ActiveTurn { Turn = t }), cancellationToken);
        var cues = new List<SubtitleCue>();
        IEnumerable<Segment> segments = transcript.Segments.IsEmpty
            ? [new Segment { Text = transcript.Text, Words = transcript.Words }]
            : transcript.Segments;
        foreach (var segment in segments)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (segment.Words.IsEmpty)
            {
                if (segment.Text.Length > 0) cues.Add(new SubtitleCue { Text = segment.Text, Start = Time(segment.Start), End = Time(segment.End), Language = segment.Language ?? transcript.Language });
                continue;
            }
            if (segment.Words.Any(w => w.Start is null || w.End is null))
            {
                var cue = Build([.. segment.Words], segment, transcript.Language);
                cues.Add(cue with { Start = Time(segment.Start).IsKnown ? Time(segment.Start) : cue.Start, End = Time(segment.End).IsKnown ? Time(segment.End) : cue.End });
                continue;
            }
            var group = new List<Word>();
            int length = 0;
            foreach (var word in segment.Words)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int separatorLength = group.Count > 0 && group[^1].Text.Length > 0 && word.Text.Length > 0 && !char.IsWhiteSpace(group[^1].Text[^1]) && !char.IsWhiteSpace(word.Text[0]) ? 1 : 0;
                if (group.Count > 0 && (Speaker(group[0]) != Speaker(word) || length + separatorLength + word.Text.Length > 84 ||
                    (group[0].Start is {} first && word.End is {} end && end - first > TimeSpan.FromSeconds(6))))
                {
                    cues.Add(Build(group, segment, transcript.Language));
                    group.Clear(); length = 0; separatorLength = 0;
                }
                group.Add(word); length += separatorLength + word.Text.Length;
            }
            if (group.Count > 0) cues.Add(Build(group, segment, transcript.Language));
        }
        // Only the display interval is made exclusive. Word spans keep their measured absolute times.
        var ordered = cues.OrderBy(c => c.Start.IsKnown ? c.Start.ToTimeSpan() : TimeSpan.MaxValue).ToList();
        for (int i = 0; i + 1 < ordered.Count; i++)
            if (ordered[i].End.IsKnown && ordered[i + 1].Start.IsKnown && ordered[i].End > ordered[i + 1].Start)
                ordered[i] = ordered[i] with { End = ordered[i + 1].Start };
        return new SubtitleDocument { Language = transcript.Language, Cues = new(ordered.Where(c => !c.Start.IsKnown || !c.End.IsKnown || c.End > c.Start)) };
    }

    private static MediaTime Time(TimeSpan? time) => time is {} value ? MediaTime.FromTimeSpan(value) : MediaTime.Unknown;
    private static string? Speaker(Word word) => word.Speaker is {} s && s.Id != SpeakerLabel.Unassigned.Id ? s.Id : null;

    private static SubtitleCue Build(List<Word> words, Segment segment, string? language)
    {
        var text = new StringBuilder();
        var spans = new List<SubtitleSpan>();
        int lineLength = 0;
        foreach (var word in words)
        {
            bool separator = text.Length > 0 && word.Text.Length > 0 && !char.IsWhiteSpace(text[^1]) && !char.IsWhiteSpace(word.Text[0]);
            if (lineLength > 0 && lineLength + word.Text.Length + (separator ? 1 : 0) > 42)
            { text.Append('\n'); lineLength = 0; separator = false; }
            if (separator) { text.Append(' '); lineLength++; }
            spans.Add(new SubtitleSpan(text.Length, word.Text.Length) { Voice = Speaker(word), Language = word.Language ?? segment.Language ?? language, WordStart = word.Start is {} start ? MediaTime.FromTimeSpan(start) : null, WordEnd = word.End is {} end ? MediaTime.FromTimeSpan(end) : null });
            text.Append(word.Text); lineLength += word.Text.Length;
        }
        var starts = words.Where(w => w.Start.HasValue).Select(w => w.Start!.Value).ToArray();
        var ends = words.Where(w => w.End.HasValue).Select(w => w.End!.Value).ToArray();
        return new SubtitleCue { Text = text.ToString(), Start = Time(starts.Length > 0 ? starts.Min() : segment.Start), End = Time(ends.Length > 0 ? ends.Max() : segment.End), Speaker = Speaker(words[0]), Language = segment.Language ?? language, Spans = new(spans) };
    }

    /// <summary>Imports cues into a Transcript. Each non-raw span becomes a word, even when untimed; cue bounds are never assigned to words. Formats may merge spans or omit untimed span boundaries. Text without spans becomes segment text. Original word confidence/provenance is not encoded by subtitle formats; imported timed words use Native with null confidence. Invalid span ranges throw ArgumentException. No speaker identity is inferred from text.</summary>
    public static Transcript ToTranscript(this SubtitleDocument document, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(document);
        cancellationToken.ThrowIfCancellationRequested();
        var segments = new List<Segment>();
        var turns = new List<Turn>();
        foreach (var cue in document.Cues)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var words = new List<Word>();
            foreach (var span in cue.Spans)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (span.Start < 0 || span.Length < 0 || span.Start > cue.Text.Length - span.Length) throw new ArgumentException("Subtitle span is outside the cue text.", nameof(document));
                if (span.Raw is not null) continue;
                string? voice = span.Voice ?? cue.Speaker;
                words.Add(new Word { Text = cue.TextOf(span), Start = Known(span.WordStart), End = Known(span.WordEnd), Speaker = voice is null ? null : new SpeakerLabel { Id = voice }, Language = span.Language ?? cue.Language ?? document.Language, TimingMethod = Known(span.WordStart) is not null || Known(span.WordEnd) is not null ? TimingMethod.Native : TimingMethod.None });
            }
            var segment = new Segment { Text = cue.Text, Start = Known(cue.Start), End = Known(cue.End), Language = cue.Language ?? document.Language, Words = [.. words] };
            segments.Add(segment);
            if (cue.Speaker is {} speaker && segment.Start is {} start && segment.End is {} end && end > start)
                turns.Add(new Turn { Text = cue.Text, Speaker = new SpeakerLabel { Id = speaker }, Start = start, End = end, Words = segment.Words });
        }
        return new Transcript { Text = string.Join("\n", segments.Select(s => s.Text)), Language = document.Language, Segments = [.. segments], Words = [.. segments.SelectMany(s => s.Words)], Turns = [.. turns], Duration = segments.Select(s => s.End ?? TimeSpan.Zero).DefaultIfEmpty().Max() };
    }

    private static TimeSpan? Known(MediaTime? time) => time is {} t && t.IsKnown ? t.ToTimeSpan() : null;

    private static SubtitleDocument TimedCaptions(Transcript transcript, CancellationToken cancellationToken)
    {
        var document = transcript.ToSubtitles(cancellationToken);
        if (document.Cues.Any(c => !c.Start.IsKnown || !c.End.IsKnown))
            throw new CapabilityNotSupportedException("Subtitle text export requires known cue bounds. Align untimed segments first, or use ToSubtitles() to retain unknown-time cues in memory.");
        return document;
    }

    /// <summary>Writes SRT text through OneMedia. Millisecond cue times and text survive; word timing and speakers cannot. Throws CapabilityNotSupportedException for unknown cue bounds; untimed word spans inside timed cues are allowed. Use ToSubtitles and SubtitleFormats.Srt.Write for its loss report.</summary>
    public static string ToSrt(this Transcript transcript, CancellationToken cancellationToken = default) => SubtitleFormats.Srt.Write(TimedCaptions(transcript, cancellationToken)).Text;
    /// <summary>Writes WebVTT through OneMedia, carrying voices and absolute word-start tags. Word ends are format-dependent. Throws CapabilityNotSupportedException for unknown cue bounds. Use the low-level writer for its loss report; cancellation is checked during conversion.</summary>
    public static string ToVtt(this Transcript transcript, CancellationToken cancellationToken = default) => SubtitleFormats.WebVtt.Write(TimedCaptions(transcript, cancellationToken)).Text;
    /// <summary>Writes ASS through OneMedia, with cue speakers and centisecond karaoke timing. Throws CapabilityNotSupportedException for unknown cue bounds. Use the low-level writer for rounding/loss reports; cancellation is checked during conversion.</summary>
    public static string ToAss(this Transcript transcript, CancellationToken cancellationToken = default) => SubtitleFormats.Ass.Write(TimedCaptions(transcript, cancellationToken)).Text;
    /// <summary>Writes TTML through OneMedia, with agents and absolute span start/end timing. Throws CapabilityNotSupportedException for unknown cue bounds. Use the low-level writer for its loss report; cancellation is checked during conversion.</summary>
    public static string ToTtml(this Transcript transcript, CancellationToken cancellationToken = default) => SubtitleFormats.Ttml.Write(TimedCaptions(transcript, cancellationToken)).Text;
}
