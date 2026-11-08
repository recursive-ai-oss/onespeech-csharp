using System.Collections.Immutable;
using OneSpeech.Audio;

namespace OneSpeech.Pipeline;

// Keep only chunk boundaries and written-word ranges, never PCM. A replay can then align
// exactly the same chunks after the much larger recognizer has released its graph weights.
internal sealed class DeferredTiming
{
    private sealed record Chunk(TimeSpan Start, TimeSpan Duration, int Offset, int Count, string? Language);
    private readonly List<Chunk> chunks = [];

    internal void Add(TimeSpan start, TimeSpan duration, int offset, int count, string? language) =>
        chunks.Add(new(start, duration, offset, count, language));

    internal Transcript Apply(IEnumerable<AudioBuffer> blocks, VoiceActivityDetector detector, Transcript transcript, Aligner aligner, CancellationToken cancellationToken)
    {
        var words = transcript.Words.ToArray();
        var models = transcript.Models.ToList();
        int index = 0;
        foreach (var audio in new VadChunker(detector).Read(blocks, null, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (index >= chunks.Count) throw new InvalidDataException("Timing replay produced extra speech chunks.");
            var plan = chunks[index++];
            if (plan.Start != audio.Start || plan.Duration != audio.Audio.Duration)
                throw new InvalidDataException("Timing replay changed a speech chunk boundary.");
            var localWords = words.AsSpan(plan.Offset, plan.Count).ToArray().Select(w => w with { Start = w.Start - plan.Start, End = w.End - plan.Start }).ToImmutableArray();
            var local = new Transcript { Text = string.Join(' ', localWords.Select(w => w.Text)), Words = localWords, Language = plan.Language, Duration = plan.Duration };
            var timed = VadTranscriptionPipeline.ApplyTiming(audio.Audio, local, plan.Language, [aligner], cancellationToken);
            for (int i = 0; i < plan.Count; i++) words[plan.Offset + i] = timed.Words[i] with { Start = timed.Words[i].Start + plan.Start, End = timed.Words[i].End + plan.Start };
            foreach (var model in timed.Models)
                if (!models.Exists(m => m.Id == model.Id && m.Revision == model.Revision)) models.Add(model);
        }
        if (index != chunks.Count) throw new InvalidDataException("Timing replay omitted speech chunks.");
        return Alignment.AlignedWords.Replace(transcript, words.ToImmutableArray()) with { Models = models.ToImmutableArray() };
    }
}
