using System.Collections.Immutable;

namespace OneSpeech.Alignment;

internal static class AlignedWords
{
    internal static Transcript Replace(Transcript original, ImmutableArray<Word> words)
    {
        if (words.Length != original.Words.Length) throw new ArgumentException("Aligned word count differs from written word count.", nameof(words));
        var references = new Dictionary<Word, Word>(ReferenceEqualityComparer.Instance);
        var unique = new Dictionary<Word, Word?>();
        for (int i = 0; i < words.Length; i++)
        {
            references.TryAdd(original.Words[i], words[i]);
            if (!unique.TryAdd(original.Words[i], words[i])) unique[original.Words[i]] = null;
        }
        Word Project(Word word) => references.TryGetValue(word, out var mapped) ? mapped :
            unique.TryGetValue(word, out var equivalent) && equivalent is not null ? equivalent :
            word with { Start = null, End = null, TimingMethod = TimingMethod.None };
        var segments = original.Segments;
        if (segments.Sum(s => s.Words.Length) == words.Length && segments.SelectMany(s => s.Words).Select(w => w.Text).SequenceEqual(original.Words.Select(w => w.Text), StringComparer.Ordinal))
        {
            int offset = 0;
            segments = segments.Select(segment =>
            {
                var projected = words.Slice(offset, segment.Words.Length);
                offset += segment.Words.Length;
                return segment with { Words = projected };
            }).ToImmutableArray();
        }
        else segments = segments.Select(s => s with { Words = s.Words.Select(Project).ToImmutableArray() }).ToImmutableArray();
        var turns = original.Turns.Select(t => t with { Words = t.Words.Select(Project).ToImmutableArray() }).ToImmutableArray();
        return original with { Words = words, Segments = segments, Turns = turns };
    }
}
