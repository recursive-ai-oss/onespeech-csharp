using System.Collections.Immutable;
using System.Globalization;
using System.Text;
using OneSpeech.Decoding;
using OneSpeech.Text;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Alignment;

/// <summary>CPU forced alignment with the published omniASR-CTC-300M character model. Instances own a OneModel session and serialize calls.</summary>
/// <remarks>Calls accept at most 60 seconds and a 16-million-cell blank-expanded target trellis. Larger audio/text combinations throw CapabilityNotSupportedException and must be split together.</remarks>
public sealed class OmniAsrAligner : Aligner, IDisposable
{
    private readonly Session session;
    private readonly Dictionary<Rune, int> characters = [];
    private readonly object gate = new();
    private bool disposed;
    /// <inheritdoc />
    public override ModelInfo Model { get; }
    /// <summary>Declared language-neutral character alignment. Language tags are retained; this model does not detect a language. English cardinal expansion is available for en/eng_Latn or an omitted hint.</summary>
    public override Capabilities Capabilities { get; } = new() { ForcedAlignment = true, Languages = ["en", "eng_Latn", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar"] };

    /// <summary>Loads converted fp32 graph and the pinned character vocabulary. Input must be mono 16 kHz PCM, at most 60 seconds per call; longer inputs must be split with corresponding text. Threads default to 1 (1–256). Throws InvalidDataException for unsupported vocabulary and OneModel loading errors for invalid graphs.</summary>
    public OmniAsrAligner(string graphPath, string tokenizerPath, ModelInfo model, int threads = 1, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (threads is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(threads));
        cancellationToken.ThrowIfCancellationRequested();
        var vocabulary = SentencePieceProto.ReadFile(tokenizerPath);
        if (vocabulary.Pieces.Length != 9812 || vocabulary.UnkId != 3 || vocabulary.Charsmap.Length != 0 || vocabulary.AddDummyPrefix || vocabulary.EscapeWhitespaces)
            throw new InvalidDataException("Expected the omniASR identity character vocabulary.");
        for (int i = 0; i < vocabulary.Pieces.Length; i++)
        {
            if (vocabulary.Types[i] != SentencePieceData.Normal) continue;
            var runes = vocabulary.Pieces[i].EnumerateRunes().ToArray();
            if (runes.Length != 1 || !characters.TryAdd(runes[0], i)) throw new InvalidDataException("Expected unique single-character pieces.");
        }
        if (!characters.ContainsKey(new Rune(' '))) throw new InvalidDataException("Character vocabulary lacks space.");
        Model = model;
        session = Session.Load(graphPath, new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 1 }, cancellationToken: cancellationToken);
    }

    /// <summary>Splits text on Unicode whitespace, retaining punctuation and casing in each written word. Empty text returns an empty word list. Speech absent or unalignable words keep null times. Maximum audio length is 60 seconds.</summary>
    public override Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(text);
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Select(w => new Word { Text = w, Language = language }).ToImmutableArray();
        return Align(audio, new Transcript { Text = text, Words = words, Language = language }, language, cancellationToken);
    }

    /// <summary>Aligns the supplied written words in their existing order without re-splitting them. All non-timing word fields and written text are preserved. Unknown graphemes and weak emissions have null times. Input is limited to mono 16 kHz, 60 s; invalid input throws ArgumentException or CapabilityNotSupportedException, and cancellation throws OperationCanceledException.</summary>
    public override Transcript Align(AudioBuffer audio, Transcript transcript, string? language, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio); ArgumentNullException.ThrowIfNull(transcript);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("CTC alignment requires mono 16000 Hz PCM.", nameof(audio));
            if (audio.Samples.Length > 60 * 16000) throw new CapabilityNotSupportedException("omniASR alignment accepts at most 60 seconds per call. Split audio with its corresponding written words.");
            if (language is not null && !Capabilities.Languages.Contains(language, StringComparer.Ordinal))
                throw new CapabilityNotSupportedException($"Alignment language '{language}' is not declared; select a declared language or omit the hint for character alignment.");
            foreach (float sample in audio.Samples.Span)
                if (!float.IsFinite(sample)) throw new ArgumentException("Audio samples must be finite.", nameof(audio));
            List<int> target = [];
            var ranges = new (int Start, int Count, bool Known)[transcript.Words.Length];
            for (int i = 0; i < transcript.Words.Length; i++)
            {
                string spoken = SpokenForm(transcript.Words[i].Text, language);
                if (spoken.Length != 0 && target.Count != 0) target.Add(characters[new Rune(' ')]);
                int start = target.Count; bool known = spoken != "\uFFFD";
                foreach (var rune in spoken.EnumerateRunes())
                {
                    if (characters.TryGetValue(rune, out int id)) target.Add(id);
                    else { target.Add(3); known = false; }
                }
                ranges[i] = (start, target.Count - start, known);
            }
            var result = transcript.Words.Select(w => w with { Start = null, End = null, TimingMethod = TimingMethod.None }).ToArray();
            if (target.Count != 0 && audio.Samples.Length >= 400)
            {
                int expectedFrames = (audio.Samples.Length - 400) / 320 + 1;
                if ((2L * target.Count + 1) * expectedFrames > 16000000)
                    throw new CapabilityNotSupportedException("CTC alignment exceeds the 16-million-cell trellis limit. Split audio with its corresponding written words.");
                using var waveform = Tensor.FromSpan<float>(audio.Samples.Span, Shape.Create(1, 1, audio.Samples.Length));
                using var output = session.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["waveform"] = waveform });
                cancellationToken.ThrowIfCancellationRequested();
                var emissions = output["logits"].Data<float>();
                if (emissions.Length != expectedFrames * 9812) throw new InvalidDataException("CTC graph returned an unexpected emission shape.");
                float[] probabilities = emissions.ToArray();
                int frames = expectedFrames;
                for (int frame = 0; frame < frames; frame++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int offset = frame * 9812;
                    float maximum = float.NegativeInfinity;
                    for (int id = 0; id < 9812; id++) maximum = Math.Max(maximum, probabilities[offset + id]);
                    double sum = 0;
                    for (int id = 0; id < 9812; id++) sum += Math.Exp(probabilities[offset + id] - maximum);
                    double normalizer = maximum + Math.Log(sum);
                    for (int id = 0; id < 9812; id++) probabilities[offset + id] = (float)(probabilities[offset + id] - normalizer);
                }
                var alignment = CtcTrellis.Align(probabilities, frames, 9812, target.ToArray(), cancellationToken: cancellationToken);
                if (alignment.Succeeded)
                {
                    for (int i = 0; i < result.Length; i++)
                    {
                        var range = ranges[i];
                        if (!range.Known || range.Count == 0) continue;
                        double score = 0; int count = 0;
                        for (int j = range.Start; j < range.Start + range.Count; j++)
                        {
                            var token = alignment.Tokens[j];
                            for (int frame = token.StartFrame; frame < token.EndFrame; frame++) { score += probabilities[frame * 9812 + token.TokenId]; count++; }
                        }
                        // Reject near-impossible constrained paths (not a calibrated word confidence).
                        if (count == 0 || score / count < Math.Log(0.01)) continue;
                        result[i] = result[i] with
                        {
                            Start = TimeSpan.FromTicks((long)alignment.Tokens[range.Start].StartFrame * 200000),
                            End = TimeSpan.FromTicks((long)alignment.Tokens[range.Start + range.Count - 1].EndFrame * 200000),
                            TimingMethod = TimingMethod.CtcAligner
                        };
                    }
                }
            }
            return AlignedWords.Replace(transcript, result.ToImmutableArray()) with { Duration = TimeSpan.FromTicks((long)audio.Samples.Length * 625), Models = [.. transcript.Models.Append(Model).DistinctBy(m => (m.Id, m.Revision))] };
        }
    }

    internal static string SpokenForm(string written, string? language)
    {
        string lower = written.Normalize(NormalizationForm.FormC).ToLowerInvariant();
        string core = lower.Trim(' ', ',', '!', '?', ':', ';', '"', '(', ')').TrimEnd('.');
        string[] groups = core.Split(',');
        bool grouping = groups.Length == 1 || (groups[0].Length is >= 1 and <= 3 && groups.Skip(1).All(g => g.Length == 3));
        bool leadingZero = core.Length > 1 && core[0] == '0';
        if ((language is null or "en" or "eng_Latn") && grouping && !leadingZero && long.TryParse(core.Replace(",", "", StringComparison.Ordinal), NumberStyles.None, CultureInfo.InvariantCulture, out long number) && number <= 999999999)
            return Cardinal(number);
        if (lower.EnumerateRunes().Any(Rune.IsDigit)) return "\uFFFD";
        StringBuilder spoken = new();
        foreach (var rune in lower.EnumerateRunes())
        {
            if (rune.Value is 0x2018 or 0x2019 or 0x02bc) spoken.Append('\'');
            else if (Rune.IsLetterOrDigit(rune) || Rune.GetUnicodeCategory(rune) is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark || rune.Value == '\'') spoken.Append(rune);
            else if (rune.Value == '-' || Rune.IsWhiteSpace(rune)) spoken.Append(' ');
        }
        return string.Join(' ', spoken.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }
    private static string Cardinal(long value)
    {
        string[] small = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
        string[] tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
        if (value < 20) return small[value];
        if (value < 100) return tens[value / 10] + (value % 10 == 0 ? "" : " " + Cardinal(value % 10));
        foreach (var (scale, name) in new (long, string)[] { (1000000, "million"), (1000, "thousand"), (100, "hundred") })
            if (value >= scale) return Cardinal(value / scale) + " " + name + (value % scale == 0 ? "" : " " + Cardinal(value % scale));
        throw new InvalidOperationException("Cardinal range error.");
    }
    /// <summary>Releases the session. Repeated disposal is harmless; subsequent alignment throws ObjectDisposedException.</summary>
    public void Dispose() { lock (gate) { if (disposed) return; disposed = true; session.Dispose(); } }
}
