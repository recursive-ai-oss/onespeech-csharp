using System.Collections.Immutable;
using System.IO.Compression;
using System.Text;
using RecursiveAI.OneModel.Graph;
using OneSpeech.Audio;
using OneSpeech.Decoding;

namespace OneSpeech.Whisper;

/// <summary>Offline Whisper recognition from independently converted local OneModel graphs and published tokenizer data.</summary>
/// <remarks>Instances are not concurrent: use one instance per transcription. Input is 16 kHz mono; file decoding and VAD are separate roles.</remarks>
public sealed class WhisperRecognizer : SpeechRecognizer, IDisposable
{
    private readonly WhisperModelSession runtime;
    private readonly WhisperTokenizer tokenizer;
    private readonly WhisperLogMelSpectrogram features;
    private readonly WhisperOptions options;
    private readonly Dictionary<string, int> languages = new(StringComparer.Ordinal);
    private readonly int start, end, transcribe, translate, previous, noSpeech, noTimestamps, timestampBegin;
    private readonly int[] suppress;
    private readonly (int Layer, int Head)[] alignmentHeads;
    private uint samplingState;
    private bool disposed;
    private WhisperRecognizer(string directory, WhisperOptions options, CancellationToken cancellationToken)
    {
        this.options = options;
        options.Validate();
        tokenizer = new WhisperTokenizer(directory);
        features = new WhisperLogMelSpectrogram(WhisperConfiguration.Load(directory).MelBins);
        using var generation = WhisperConfiguration.ReadJson(Path.Combine(directory, "generation_config.json"));
        var g = generation.RootElement;
        start = g.GetProperty("decoder_start_token_id").GetInt32();
        end = g.GetProperty("eos_token_id").GetInt32();
        transcribe = g.GetProperty("task_to_id").GetProperty("transcribe").GetInt32();
        translate = g.GetProperty("task_to_id").GetProperty("translate").GetInt32();
        previous = tokenizer.GetToken("<|startofprev|>");
        noSpeech = tokenizer.TryGetToken("<|nospeech|>", out int noSpeechId) ? noSpeechId : tokenizer.GetToken("<|nocaptions|>");
        noTimestamps = g.GetProperty("no_timestamps_token_id").GetInt32();
        timestampBegin = noTimestamps + 1;
        foreach (var entry in g.GetProperty("lang_to_id").EnumerateObject()) languages.Add(entry.Name[2..^2], entry.Value.GetInt32());
        suppress = g.GetProperty("suppress_tokens").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        alignmentHeads = g.TryGetProperty("alignment_heads", out var heads) ? heads.EnumerateArray().Select(v => (v[0].GetInt32(), v[1].GetInt32())).ToArray() : [];
        string[] source = WhisperCheckpoint.Read(directory);
        Model = WhisperCheckpoint.Provenance(source);
        Capabilities = new Capabilities { Languages = languages.Keys.ToImmutableArray(), Translation = !source[0].EndsWith("turbo", StringComparison.Ordinal), NativeTimestamps = false, Streaming = false, Biasing = false };
        if (options.Task == WhisperTask.Translate && !Capabilities.Translation) throw new CapabilityNotSupportedException("This checkpoint does not support translation.");
        runtime = new WhisperModelSession(directory, options.Threads, cancellationToken);
    }

    /// <summary>Loads local encoder/decoder OneModel graphs plus config, generation config, vocab, merges, added tokens and upstream provenance. Does not access the network.</summary>
    /// <exception cref="IOException">A required file is missing, malformed or unreadable.</exception>
    /// <exception cref="ArgumentException">Options are invalid.</exception>
    /// <exception cref="CapabilityNotSupportedException">The checkpoint cannot serve the requested translation task.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested while loading.</exception>
    public static WhisperRecognizer Load(string directory, WhisperOptions? options = null, CancellationToken cancellationToken = default) => new(directory, options ?? new WhisperOptions(), cancellationToken);

    /// <inheritdoc/>
    public override ModelInfo Model { get; }
    /// <inheritdoc/>
    public override Capabilities Capabilities { get; }

    /// <inheritdoc/>
    public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        var segments = TranscribeStream(audio, language, cancellationToken).ToImmutableArray();
        return new Transcript { Text = string.Concat(segments.Select(s => s.Text)), Segments = segments, Words = segments.SelectMany(s => s.Words).ToImmutableArray(), Duration = audio.Duration, Language = language ?? segments.FirstOrDefault()?.Language, Models = [Model] };
    }

    /// <inheritdoc/>
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        ValidateAudio(audio);
        cancellationToken.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(disposed, this);
        if (language != null && !languages.ContainsKey(language)) throw new CapabilityNotSupportedException("Unsupported Whisper language: " + language);
        samplingState = options.SamplingSeed ^ 0x9e3779b9u;
        float fileFloor = float.NegativeInfinity;
        if (audio.FrameCount > WhisperLogMelSpectrogram.WindowSamples)
            for (int position = 0; position < audio.FrameCount; position += WhisperLogMelSpectrogram.WindowSamples)
                fileFloor = MathF.Max(fileFloor, features.ExtractContext(audio, position, float.NegativeInfinity, cancellationToken).Max() - 2);
        List<int> context = [];
        int seek = 0;
        while (seek < audio.FrameCount)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(WhisperLogMelSpectrogram.WindowSamples, audio.FrameCount - seek);
            using var encoded = runtime.Encode(features.ExtractContext(audio, seek, fileFloor, cancellationToken), cancellationToken);
            language ??= DetectLanguage(encoded, cancellationToken);
            List<int> prefix = [];
            if (context.Count > 0) { prefix.Add(previous); prefix.AddRange(context.TakeLast(223)); }
            prefix.Add(start); prefix.Add(languages[language]); prefix.Add(options.Task == WhisperTask.Transcribe ? transcribe : translate);
            Attempt result = Decode(encoded, prefix.ToArray(), cancellationToken);
            bool skip = options.NoSpeechThreshold.HasValue && result.NoSpeech > options.NoSpeechThreshold.Value && !(options.LogProbabilityThreshold.HasValue && result.AverageLogProbability > options.LogProbabilityThreshold.Value);
            if (skip) { seek += count; options.Progress?.Invoke(TimeSpan.FromSeconds(seek / 16000.0)); continue; }
            var sliced = Slice(result.Tokens, seek, count, language);
            int advance = sliced.Advance;
            if (options.WordTimestamps && sliced.Segments.Count > 0)
            {
                var combined = new Segment { Text = string.Concat(sliced.Segments.Select(s => s.Text)), Language = language };
                var alignment = AddWordTimes(encoded, combined, seek, count, language, cancellationToken, sliced.TextTokens);
                List<Word>[] groups = sliced.Segments.Select(_ => new List<Word>()).ToArray();
                int group = 0, position = 0, boundary = sliced.Segments[0].Text.Length;
                foreach (Word word in alignment.Words)
                {
                    while (group + 1 < groups.Length && position >= boundary) { group++; boundary += sliced.Segments[group].Text.Length; }
                    groups[group].Add(word); position += word.Text.Length;
                }
                for (int i = 0; i < groups.Length; i++) sliced.Segments[i] = sliced.Segments[i] with { Words = groups[i].ToImmutableArray() };
            }
            foreach (Segment segment in sliced.Segments)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Segment timed = options.WordTimestamps ? segment : AddUntimedWords(segment);
                yield return timed;
            }
            if (!options.ConditionOnPreviousText || result.Temperature > .5) context.Clear();
            else { context.AddRange(result.Tokens.Take(sliced.UsedTokens)); if (context.Count > 223) context.RemoveRange(0, context.Count - 223); }
            seek += Math.Clamp(advance, 1, count);
            options.Progress?.Invoke(TimeSpan.FromSeconds(seek / 16000.0));
        }
    }

    /// <inheritdoc/>
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken)
    {
        ValidateAudio(audio);
        cancellationToken.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(disposed, this);
        int count = Math.Min(audio.FrameCount, WhisperLogMelSpectrogram.WindowSamples);
        using var encoded = runtime.Encode(features.Extract(new AudioBuffer(audio.Samples[..count], 16000), cancellationToken: cancellationToken), cancellationToken);
        return DetectLanguage(encoded, cancellationToken);
    }
    private string DetectLanguage(SessionOutputs encoded, CancellationToken ct)
    {
        using var decoded = runtime.Decode(encoded, new[] { start }, null, 0, ct);
        var logits = decoded["logits"].Data<float>();
        string best = ""; float score = float.NegativeInfinity;
        foreach (var language in languages) if (logits[language.Value] > score) { score = logits[language.Value]; best = language.Key; }
        return best;
    }
    private static void ValidateAudio(AudioBuffer audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Whisper expects 16 kHz mono PCM.", nameof(audio));
    }

    internal int[] DecodeProbe(float[] mel, string language, CancellationToken cancellationToken)
    {
        using var encoded = runtime.Encode(mel, cancellationToken);
        return Decode(encoded, [start, languages[language], options.Task == WhisperTask.Transcribe ? transcribe : translate], cancellationToken).Tokens;
    }
    private sealed record Attempt(int[] Tokens, double AverageLogProbability, double NoSpeech, double Temperature);
    private sealed class DecoderBeamState(WhisperModelSession model, SessionOutputs encoded, SessionOutputs decoded, int length) : IBeamDecoderState
    {
        private readonly float[] logits = decoded["logits"].Data<float>()[^model.Configuration.Vocabulary..].ToArray();
        public ReadOnlyMemory<float> Logits => logits;
        public IBeamDecoderState Advance(int token, CancellationToken cancellationToken) => new DecoderBeamState(model, encoded, model.Decode(encoded, new[] { token }, decoded, length, cancellationToken), length + 1);
        public void Dispose() => decoded.Dispose();
    }
    private Attempt DecodeBeam(SessionOutputs encoded, int[] prefix, CancellationToken cancellationToken)
    {
        var decoded = runtime.Decode(encoded, prefix, null, 0, cancellationToken);
        int vocab = runtime.Configuration.Vocabulary;
        var startLogits = decoded["logits"].Data<float>().Slice(Array.IndexOf(prefix, start) * vocab, vocab);
        double noSpeechProbability = Math.Exp(startLogits[noSpeech] - LogSumExp(startLogits));
        var root = new DecoderBeamState(runtime, encoded, decoded, prefix.Length);
        var result = BeamSearch.Decode(root, end, options.BeamSize!.Value, Math.Min(options.MaxTokens, 448 - prefix.Length), (logits, tokens) => Filter(logits, tokens.ToArray().ToList(), tokens.Length == 0), cancellationToken);
        return new Attempt(result.Tokens.ToArray(), result.MeanLogProbability, noSpeechProbability, 0);
    }
    private bool NeedsFallback(Attempt result)
    {
        bool fallback = options.CompressionRatioThreshold.HasValue && CompressionRatio(tokenizer.Decode(result.Tokens)) > options.CompressionRatioThreshold.Value || options.LogProbabilityThreshold.HasValue && result.AverageLogProbability < options.LogProbabilityThreshold.Value;
        if (options.NoSpeechThreshold.HasValue && result.NoSpeech > options.NoSpeechThreshold.Value && options.LogProbabilityThreshold.HasValue && result.AverageLogProbability < options.LogProbabilityThreshold.Value) fallback = false;
        return fallback;
    }
    private Attempt Decode(SessionOutputs encoded, int[] prefix, CancellationToken ct)
    {
        Attempt? last = null;
        foreach (double temperature in options.Temperatures)
        {
            if (temperature == 0 && options.BeamSize.HasValue)
            {
                last = DecodeBeam(encoded, prefix, ct);
                if (!NeedsFallback(last)) return last;
                continue;
            }
            List<int> generated = [];
            SessionOutputs? state = null;
            double sumLogProbability = 0, noSpeechProbability = 0;
            try
            {
                int past = 0;
                for (int step = 0; step < options.MaxTokens && prefix.Length + generated.Count < 448; step++)
                {
                    SessionOutputs next = runtime.Decode(encoded, step == 0 ? prefix : new[] { generated[^1] }, state, past, ct);
                    state?.Dispose(); state = next;
                    past += step == 0 ? prefix.Length : 1;
                    var allLogits = state["logits"].Data<float>();
                    int vocab = runtime.Configuration.Vocabulary;
                    if (step == 0)
                    {
                        int sotPosition = Array.IndexOf(prefix, start);
                        var startLogits = allLogits.Slice(sotPosition * vocab, vocab);
                        noSpeechProbability = Math.Exp(startLogits[noSpeech] - LogSumExp(startLogits));
                    }
                    float[] logits = allLogits[^vocab..].ToArray();
                    Filter(logits, generated, step == 0);
                    double normalizer = LogSumExp(logits);
                    int selected = Select(logits, temperature);
                    sumLogProbability += logits[selected] - normalizer;
                    if (selected == end) break;
                    generated.Add(selected);
                }
            }
            finally { state?.Dispose(); }
            double average = sumLogProbability / (generated.Count + 1);
            last = new(generated.ToArray(), average, noSpeechProbability, temperature);
            if (!NeedsFallback(last)) return last;
        }
        return last!;
    }

    private void Filter(float[] logits, List<int> generated, bool initial)
    {
        foreach (int token in suppress) if ((uint)token < logits.Length) logits[token] = float.NegativeInfinity;
        foreach (int token in new[] { start, transcribe, translate, previous, noSpeech, noTimestamps, tokenizer.GetToken("<|startoflm|>") }) logits[token] = float.NegativeInfinity;
        if (initial)
        {
            logits[end] = float.NegativeInfinity;
            foreach (int token in tokenizer.Encode(" ")) logits[token] = float.NegativeInfinity;
        }
        bool lastTimestamp = generated.Count > 0 && generated[^1] >= timestampBegin;
        bool previousTimestamp = generated.Count < 2 || generated[^2] >= timestampBegin;
        if (lastTimestamp)
        {
            if (previousTimestamp) Array.Fill(logits, float.NegativeInfinity, timestampBegin, logits.Length - timestampBegin);
            else Array.Fill(logits, float.NegativeInfinity, 0, end);
        }
        int last = generated.FindLast(t => t >= timestampBegin);
        if (last >= timestampBegin)
        {
            int limit = lastTimestamp && !previousTimestamp ? last : last + 1;
            Array.Fill(logits, float.NegativeInfinity, timestampBegin, Math.Min(limit, logits.Length) - timestampBegin);
        }
        if (initial)
        {
            Array.Fill(logits, float.NegativeInfinity, 0, timestampBegin);
            int begin = timestampBegin + (int)Math.Round(options.MaxInitialTimestamp / .02) + 1;
            if (begin < logits.Length) Array.Fill(logits, float.NegativeInfinity, begin, logits.Length - begin);
        }
        double timestamps = LogSumExp(logits.AsSpan(timestampBegin));
        float bestText = float.NegativeInfinity;
        for (int i = 0; i < timestampBegin; i++) bestText = Math.Max(bestText, logits[i]);
        if (timestamps > bestText) Array.Fill(logits, float.NegativeInfinity, 0, timestampBegin);
    }
    private int Select(float[] logits, double temperature)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        if (!float.IsFinite(logits[best])) throw new InvalidDataException("No permitted Whisper token remains.");
        if (temperature == 0) return best;
        double total = 0;
        foreach (float value in logits) total += Math.Exp((value - logits[best]) / temperature);
        // Explicit deterministic xorshift32 state: reproducible sampling without a process-global RNG.
        samplingState ^= samplingState << 13; samplingState ^= samplingState >> 17; samplingState ^= samplingState << 5;
        double target = (samplingState + .5) / 4294967296.0 * total;
        for (int i = 0; i < logits.Length; i++) { target -= Math.Exp((logits[i] - logits[best]) / temperature); if (target <= 0) return i; }
        return best;
    }
    internal static double LogSumExp(ReadOnlySpan<float> values)
    {
        float max = float.NegativeInfinity; foreach (float value in values) max = Math.Max(max, value);
        if (float.IsNegativeInfinity(max)) return double.NegativeInfinity;
        double sum = 0; foreach (float value in values) sum += Math.Exp(value - max);
        return max + Math.Log(sum);
    }
    private static double CompressionRatio(string text)
    {
        byte[] input = Encoding.UTF8.GetBytes(text);
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Optimal, leaveOpen: true)) zlib.Write(input);
        return (double)input.Length / output.Length;
    }

    private (List<Segment> Segments, int Advance, int UsedTokens, int[] TextTokens) Slice(int[] tokens, int offset, int count, string language)
    {
        List<int> cuts = [];
        for (int i = 1; i < tokens.Length; i++) if (tokens[i - 1] >= timestampBegin && tokens[i] >= timestampBegin) cuts.Add(i);
        bool singleEnding = tokens.Length >= 2 && tokens[^1] >= timestampBegin && tokens[^2] < timestampBegin;
        if (cuts.Count > 0 && singleEnding) cuts.Add(tokens.Length);
        List<Segment> segments = [];
        List<int> selectedTextTokens = [];
        int begin = 0;
        foreach (int cut in cuts)
        {
            int[] slice = tokens[begin..cut];
            int[] timestamps = slice.Where(t => t >= timestampBegin).ToArray();
            if (timestamps.Length >= 2) Add(slice, (timestamps[0] - timestampBegin) * 320, (timestamps[^1] - timestampBegin) * 320);
            begin = cut;
        }
        if (cuts.Count == 0)
        {
            int endpoint = tokens.LastOrDefault(t => t >= timestampBegin, timestampBegin);
            Add(tokens, 0, endpoint > timestampBegin ? (endpoint - timestampBegin) * 320 : count);
            return (segments, count, tokens.Length, selectedTextTokens.ToArray());
        }
        int advance = singleEnding ? count : (tokens[begin - 1] - timestampBegin) * 320;
        return (segments, advance, begin, selectedTextTokens.ToArray());
        void Add(int[] slice, int startSamples, int endSamples)
        {
            string text = tokenizer.Decode(slice);
            if (string.IsNullOrWhiteSpace(text) || endSamples <= startSamples) return;
            selectedTextTokens.AddRange(slice.Where(t => t < end));
            segments.Add(new Segment { Text = text, Start = TimeSpan.FromSeconds((offset + Math.Min(startSamples, count)) / 16000.0), End = TimeSpan.FromSeconds((offset + Math.Min(endSamples, count)) / 16000.0), Language = language });
        }
    }

    private static Segment AddUntimedWords(Segment segment) => segment with { Words = segment.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(t => new Word { Text = t, Language = segment.Language }).ToImmutableArray() };
    private Segment AddWordTimes(SessionOutputs encoded, Segment segment, int offset, int count, string language, CancellationToken cancellationToken, int[]? originalTokens = null)
        => WhisperWordAlignment.Align(runtime, tokenizer, alignmentHeads, encoded, segment, offset, count, [start, languages[language], options.Task == WhisperTask.Transcribe ? transcribe : translate, noTimestamps], end, cancellationToken, originalTokens);

    /// <summary>Releases model sessions. Subsequent processing throws ObjectDisposedException.</summary>
    public void Dispose() { if (disposed) return; runtime.Dispose(); disposed = true; }
}
