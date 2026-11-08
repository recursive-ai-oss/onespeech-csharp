using System.Collections.Immutable;
using System.Text;
using OneSpeech.Decoding;
using RecursiveAI.OneModel.Graph;

namespace OneSpeech.Whisper;

internal static class WhisperWordAlignment
{
    internal static Segment Align(WhisperModelSession model, WhisperTokenizer tokenizer, (int Layer, int Head)[] heads, SessionOutputs encoded, Segment segment, int offset, int count, int[] prefix, int endToken, CancellationToken ct)
    {
        int[] text = tokenizer.Encode(segment.Text, cancellationToken: ct);
        var words = SplitWords(tokenizer, text);
        Segment Untimed() => segment with { Words = words.Select(w => new Word { Text = w.Text, Language = segment.Language }).ToImmutableArray() };
        int frames = Math.Min(1500, count / 320);
        if (heads.Length == 0 || text.Length == 0 || prefix.Length + text.Length + 1 > 448 || frames <= 0) return Untimed();
        int[] sequence = [.. prefix, .. text, endToken];
        using var decoded = model.Decode(encoded, sequence, null, 0, ct);
        int rows = text.Length + 1;
        float[] costs = new float[rows * frames];
        float[] normalized = new float[rows * frames];
        float[] window = new float[7];
        foreach (var head in heads)
        {
            ct.ThrowIfCancellationRequested();
            if (head.Layer >= model.Configuration.DecoderLayers || head.Head >= model.Configuration.DecoderHeads) return Untimed();
            var attention = decoded["attention_" + head.Layer].Data<float>();
            int begin = head.Head * sequence.Length * 1500 + (prefix.Length - 1) * 1500;
            for (int f = 0; f < frames; f++)
            {
                double sum = 0, squares = 0;
                for (int r = 0; r < rows; r++) { double value = attention[begin + r * 1500 + f]; sum += value; squares += value * value; }
                double mean = sum / rows, std = Math.Sqrt(Math.Max(0, squares / rows - mean * mean));
                if (std < 1e-20) std = 1;
                for (int r = 0; r < rows; r++) normalized[r * frames + f] = (float)((attention[begin + r * 1500 + f] - mean) / std);
            }
            for (int r = 0; r < rows; r++)
                for (int f = 0; f < frames; f++)
                {
                    for (int k = -3; k <= 3; k++)
                    {
                        int index = f + k;
                        if (index < 0) index = -index;
                        if (index >= frames) index = 2 * frames - 2 - index;
                        index = Math.Clamp(index, 0, frames - 1);
                        window[k + 3] = normalized[r * frames + index];
                    }
                    Array.Sort(window);
                    costs[r * frames + f] -= window[3] / heads.Length;
                }
        }
        if (costs.Any(v => !float.IsFinite(v))) return Untimed();
        var path = DynamicTimeWarping.Align(costs, rows, frames, ct);
        int[] boundaries = new int[rows];
        Array.Fill(boundaries, -1);
        foreach (var point in path) if (boundaries[point.Row] < 0) boundaries[point.Row] = point.Column;
        if (boundaries.Any(v => v < 0)) return Untimed();
        var logits = decoded["logits"].Data<float>();
        double[] probabilities = new double[text.Length];
        int vocab = model.Configuration.Vocabulary;
        for (int t = 0; t < text.Length; t++)
        {
            var values = logits.Slice((prefix.Length - 1 + t) * vocab, vocab);
            probabilities[t] = Math.Exp(values[text[t]] - WhisperRecognizer.LogSumExp(values));
        }
        List<Word> aligned = [];
        foreach (var word in words)
        {
            int begin = boundaries[word.Begin], end = boundaries[word.End];
            double confidence = probabilities.AsSpan(word.Begin, word.End - word.Begin).ToArray().Average();
            aligned.Add(new Word { Text = word.Text, Start = TimeSpan.FromSeconds((offset + begin * 320) / 16000.0), End = TimeSpan.FromSeconds((offset + end * 320) / 16000.0), Confidence = Math.Clamp(confidence, 0, 1), TimingMethod = TimingMethod.Dtw, Language = segment.Language });
        }
        return segment with { Words = aligned.ToImmutableArray() };
    }

    private sealed record TokenWord(string Text, int Begin, int End);
    private static List<TokenWord> SplitWords(WhisperTokenizer tokenizer, int[] tokens)
    {
        List<TokenWord> words = [];
        int begin = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            string piece = tokenizer.Decode(tokens.AsSpan(begin, i - begin + 1));
            if (piece.Contains('\ufffd', StringComparison.Ordinal)) continue;
            bool punctuation = piece.Trim().EnumerateRunes().All(r => Rune.IsPunctuation(r));
            if (words.Count == 0 || piece.StartsWith(' ') || punctuation)
                words.Add(new TokenWord(piece, begin, i + 1));
            else words[^1] = words[^1] with { Text = words[^1].Text + piece, End = i + 1 };
            begin = i + 1;
        }
        if (begin < tokens.Length) words.Add(new TokenWord(tokenizer.Decode(tokens.AsSpan(begin)), begin, tokens.Length));
        const string prepend = "\"'“¿([{-";
        const string append = "\"'.。,，!！?？:：”)]}、";
        for (int i = 0; i + 1 < words.Count;)
        {
            string text = words[i].Text;
            if (text.StartsWith(' ') && text.Trim().Length == 1 && prepend.Contains(text.Trim(), StringComparison.Ordinal))
            {
                words[i + 1] = words[i + 1] with { Text = text + words[i + 1].Text, Begin = words[i].Begin }; words.RemoveAt(i);
            }
            else i++;
        }
        for (int i = 1; i < words.Count;)
        {
            if (words[i].Text.Trim().Length == 1 && append.Contains(words[i].Text.Trim(), StringComparison.Ordinal) && !words[i - 1].Text.EndsWith(' '))
            {
                words[i - 1] = words[i - 1] with { Text = words[i - 1].Text + words[i].Text, End = words[i].End }; words.RemoveAt(i);
            }
            else i++;
        }
        return words;
    }
}
