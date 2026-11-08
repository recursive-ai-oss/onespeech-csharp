using System.Collections.Immutable;

namespace OneSpeech.Decoding;

/// <summary>A written word and its contiguous nonblank-token range, for tokenizer-independent native timing.</summary>
/// <param name="Text">Written form, preserved without normalization.</param>
/// <param name="FirstToken">Zero-based index into decoded tokens.</param>
/// <param name="TokenCount">Number of tokens; zero represents an untimed word.</param>
public readonly record struct WordTokenRange(string Text, int FirstToken, int TokenCount);

/// <summary>Maps encoder-frame alignments onto the audio clock without interpolation or invented word lengths.</summary>
public static class NativeWordTiming
{
    /// <summary>Unions each word's token intervals and averages token confidences. Zero-token words have null timing and confidence.
    /// Overlapping token intervals and zero durations are preserved. Returned words are not assigned to a speaker.</summary>
    /// <param name="tokens">Nonblank tokens in decoding order, with nondecreasing starts and end at least start.</param>
    /// <param name="words">Written words and token ranges, ordered and nonoverlapping. Gaps are allowed.</param>
    /// <param name="frameDuration">Positive encoder frame duration; for 8x subsampling of 10 ms features, 80 ms.</param>
    /// <param name="offset">Nonnegative audio-clock chunk offset; zero by default.</param>
    /// <param name="cancellationToken">Checked per word.</param>
    /// <exception cref="ArgumentException">A range, time scale, interval or confidence is invalid.</exception>
    /// <exception cref="OverflowException">Times exceed TimeSpan capacity.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static ImmutableArray<Word> Map(ReadOnlySpan<FrameToken> tokens, ReadOnlySpan<WordTokenRange> words, TimeSpan frameDuration, TimeSpan offset = default, CancellationToken cancellationToken = default)
    {
        if (frameDuration <= TimeSpan.Zero || offset < TimeSpan.Zero) throw new ArgumentException("Invalid audio time scale.");
        cancellationToken.ThrowIfCancellationRequested();
        int previousStart = 0;
        foreach (var token in tokens)
        {
            if (token.Id < 0 || token.StartFrame < previousStart || token.EndFrame < token.StartFrame || !float.IsFinite(token.Confidence) || token.Confidence is < 0 or > 1)
                throw new ArgumentException("Invalid token alignment.", nameof(tokens));
            previousStart = token.StartFrame;
        }
        var result = ImmutableArray.CreateBuilder<Word>(words.Length);
        int previousEnd = 0;
        foreach (var word in words)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (word.Text is null || word.FirstToken < previousEnd || word.TokenCount < 0 || word.FirstToken > tokens.Length - word.TokenCount)
                throw new ArgumentException("Invalid word token range.", nameof(words));
            previousEnd = word.FirstToken + word.TokenCount;
            if (word.TokenCount == 0) { result.Add(new() { Text = word.Text }); continue; }
            var range = tokens.Slice(word.FirstToken, word.TokenCount);
            int end = range[0].EndFrame;
            double sum = 0;
            foreach (var token in range) { end = Math.Max(end, token.EndFrame); sum += token.Confidence; }
            result.Add(new()
            {
                Text = word.Text,
                Start = TimeSpan.FromTicks(checked(offset.Ticks + frameDuration.Ticks * range[0].StartFrame)),
                End = TimeSpan.FromTicks(checked(offset.Ticks + frameDuration.Ticks * end)),
                Confidence = sum / range.Length,
                TimingMethod = TimingMethod.Native
            });
        }
        return result.MoveToImmutable();
    }
}
