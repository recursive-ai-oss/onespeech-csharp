using System.Buffers.Binary;
using System.Collections.Immutable;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Decoding;
using OneSpeech.Pipeline;
using OneSpeech.Vad;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;

namespace OneSpeech.FastConformer;

/// <summary>Offline greedy fp32 Parakeet-TDT-v3 recognizer with native 80 ms frame timings. Instances are not thread-safe.</summary>
/// <remarks>Loads only explicitly selected tier-2 cache files. Audio is mono 16 kHz; input longer than 30 seconds uses VAD-first chunks. No language conditioning, translation, phrase biasing or language-ID head is provided.</remarks>
public sealed class ParakeetRecognizer : SpeechRecognizer, IDisposable
{
    private readonly Model encoderFirst;
    private readonly Model encoderSecond;
    private readonly Model predictor;
    private readonly Model joint;
    private readonly float[] embeddings;
    private readonly ParakeetVocabulary vocabulary;
    private readonly NemoLogMelSpectrogram features = new();
    private bool disposed;

    private ParakeetRecognizer(ResolvedModel resolved, int threads, CancellationToken cancellationToken)
    {
        if (resolved.Info.Id != "nvidia/parakeet-tdt-0.6b-v3" || resolved.Info.Revision != "541d1f99c6b0c3cd0b11a95167540bb8edefd82b" || resolved.Info.Tier != ModelTier.Tier2)
            throw new ModelIntegrityException("Expected the pinned publisher Parakeet-TDT-v3 model.");
        Model = resolved.Info;
        string directory = resolved.Directory;
        vocabulary = new(Path.Combine(directory, "tokenizer.json"));
        using (var file = File.OpenRead(Path.Combine(directory, "embedding.f32")))
        {
            if (file.Length != 8193L * 640 * 4) throw new ModelIntegrityException("Invalid Parakeet embedding size.");
            embeddings = new float[8193 * 640]; byte[] buffer = new byte[640 * 4];
            for (int row = 0; row < 8193; row++)
            {
                cancellationToken.ThrowIfCancellationRequested(); file.ReadExactly(buffer);
                for (int column = 0; column < 640; column++) embeddings[row * 640 + column] = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(column * 4));
            }
        }
        var options = new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 2 };
        Model? first = null, second = null, pred = null, join = null;
        try
        {
            first = RecursiveAI.OneModel.Model.Load(Path.Combine(directory, "encoder-part0.onemodel"), options, cancellationToken: cancellationToken);
            second = RecursiveAI.OneModel.Model.Load(Path.Combine(directory, "encoder-part1.onemodel"), options, cancellationToken: cancellationToken);
            pred = RecursiveAI.OneModel.Model.Load(Path.Combine(directory, "predictor.onemodel"), options, cancellationToken: cancellationToken);
            join = RecursiveAI.OneModel.Model.Load(Path.Combine(directory, "joint.onemodel"), options, cancellationToken: cancellationToken);
            encoderFirst = first; encoderSecond = second; predictor = pred; joint = join;
        }
        catch { first?.Dispose(); second?.Dispose(); pred?.Dispose(); join?.Dispose(); throw; }
    }

    /// <summary>Loads a publisher-verified, locally converted resolved cache entry. Threads default to zero (runtime selection); nonnegative values select the intra-op limit.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Threads is negative.</exception>
    /// <exception cref="ModelIntegrityException">Provenance or an artifact is invalid.</exception>
    /// <exception cref="IOException">A required local artifact is missing or unreadable.</exception>
    /// <exception cref="OperationCanceledException">Loading is canceled.</exception>
    public static ParakeetRecognizer Load(ResolvedModel model, int threads = 0, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        cancellationToken.ThrowIfCancellationRequested();
        return new(model, threads, cancellationToken);
    }

    /// <summary>Pinned publisher provenance, licence and attribution, including locally converted file digests.</summary>
    public override ModelInfo Model { get; }
    /// <summary>The 25 supported languages and native timing; no true streaming, biasing, translation or independent language detection.</summary>
    public override Capabilities Capabilities => new()
    {
        Languages = ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"],
        NativeTimestamps = true
    };

    /// <summary>Transcribes finite mono 16 kHz PCM. Language, if supplied, must be in the supported set and labels the output; recognition itself always selects language acoustically. Sub-20 ms input returns no words. Long audio uses 30-second VAD chunks.</summary>
    /// <exception cref="ObjectDisposedException">The recognizer has been disposed.</exception>
    /// <exception cref="ArgumentException">PCM is not mono 16 kHz or contains nonfinite samples.</exception>
    /// <exception cref="CapabilityNotSupportedException">Language is unsupported.</exception>
    /// <exception cref="OperationCanceledException">Inference is canceled at graph or decoding steps.</exception>
    public override Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this); ArgumentNullException.ThrowIfNull(audio);
        cancellationToken.ThrowIfCancellationRequested();
        if (language != null && !Capabilities.Languages.Contains(language)) throw new CapabilityNotSupportedException("Parakeet-TDT-v3 does not support language '" + language + "'.");
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Parakeet requires mono 16 kHz PCM.", nameof(audio));
        if (audio.FrameCount > 480000)
        {
            using var vad = new SileroVad(cancellationToken: cancellationToken);
            return new VadTranscriptionPipeline(this, vad).Transcribe(audio, language, cancellationToken);
        }
        if (audio.FrameCount < 320)
        {
            foreach (float sample in audio.Samples.Span) if (!float.IsFinite(sample)) throw new ArgumentException("PCM must be finite.", nameof(audio));
            return new Transcript { Text = "", Language = language, Duration = audio.Duration, Models = [Model] };
        }
        return TranscribeDecoded(audio, Decode(audio, cancellationToken), language, cancellationToken);
    }

    internal Transcript TranscribeDecoded(AudioBuffer audio, ImmutableArray<FrameToken> tokens, string? language, CancellationToken cancellationToken) => CreateTranscript(audio, tokens, vocabulary, Model, language, cancellationToken);

    internal static Transcript CreateTranscript(AudioBuffer audio, ImmutableArray<FrameToken> tokens, ParakeetVocabulary vocabulary, ModelInfo model, string? language, CancellationToken cancellationToken)
    {
        var words = NativeWordTiming.Map(vocabulary.TimingTokens(tokens.AsSpan()), vocabulary.TimingRanges(tokens.AsSpan(), vocabulary.Words(tokens.AsSpan())), TimeSpan.FromMilliseconds(80), cancellationToken: cancellationToken);
        // A final frame can extend past the final PCM sample; clip native intervals to the supplied audio horizon.
        words = words.Select(word => word with { Start = word.Start > audio.Duration ? audio.Duration : word.Start, End = word.End > audio.Duration ? audio.Duration : word.End, Language = language }).ToImmutableArray();
        string text = vocabulary.Decode(tokens.Select(token => token.Id).ToArray());
        var segment = new Segment { Start = TimeSpan.Zero, End = audio.Duration, Text = text, Words = words, Language = language };
        return new Transcript { Text = text, Language = language, Duration = audio.Duration, Models = [model], Words = words, Segments = [segment] };
    }

    internal ImmutableArray<FrameToken> Decode(AudioBuffer audio, CancellationToken cancellationToken)
    {
        float[] padded = features.Extract(audio, cancellationToken: cancellationToken);
        int frames = NemoLogMelSpectrogram.GetFrameCount(audio.FrameCount);
        float[] mel = new float[128 * frames];
        // The reference encoder crops the masked trailing STFT frame before subsampling (observed on odd frame lengths).
        for (int channel = 0; channel < 128; channel++) padded.AsSpan(channel * (frames + 1), frames).CopyTo(mel.AsSpan(channel * frames));
        int encodedFrames = (frames + 7) / 8;
        float[] encoded = Run(encoderSecond, Run(encoderFirst, mel, [1, 128, frames], cancellationToken), [1, 1024, encodedFrames], cancellationToken);
        float[] state = Run(predictor, new float[3200], [1, 3200], cancellationToken);
        float[] jointInput = new float[1664];
        int accepted = 0;
        void Logits(int frame, ReadOnlySpan<int> prefix, Span<float> tokenLogits, Span<float> durationLogits, CancellationToken ct)
        {
            while (accepted < prefix.Length)
            {
                embeddings.AsSpan(prefix[accepted++] * 640, 640).CopyTo(state);
                state = Run(predictor, state, [1, 3200], ct);
            }
            for (int channel = 0; channel < 1024; channel++) jointInput[channel] = encoded[channel * encodedFrames + frame];
            state.AsSpan(0, 640).CopyTo(jointInput.AsSpan(1024));
            float[] logits = Run(joint, jointInput, [1, 1664], ct);
            logits.AsSpan(0, 8193).CopyTo(tokenLogits); logits.AsSpan(8193, 5).CopyTo(durationLogits);
        }
        return TdtGreedy.Decode(encodedFrames, 8193, 8192, [0, 1, 2, 3, 4], Logits, maxSymbolsPerFrame: 10, cancellationToken: cancellationToken);
    }

    private static float[] Run(Model model, float[] input, int[] shape, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        float[] result = model.Run(input, shape);
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    /// <summary>Yields immutable final segments with VAD-first 30-second chunks. Input, language, disposal and cancellation rules match Transcribe.</summary>
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this); ArgumentNullException.ThrowIfNull(audio);
        cancellationToken.ThrowIfCancellationRequested();
        if (language != null && !Capabilities.Languages.Contains(language)) throw new CapabilityNotSupportedException("Parakeet-TDT-v3 does not support language '" + language + "'.");
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Parakeet requires mono 16 kHz PCM.", nameof(audio));
        using var vad = new SileroVad(cancellationToken: cancellationToken);
        foreach (var segment in new VadTranscriptionPipeline(this, vad).TranscribeStream(audio, language, cancellationToken)) yield return segment;
    }

    /// <summary>Throws CapabilityNotSupportedException: this checkpoint does not expose an independent language-ID result.</summary>
    public override string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this); cancellationToken.ThrowIfCancellationRequested();
        throw new CapabilityNotSupportedException("Parakeet-TDT-v3 has no separate language-ID output; use Whisper or Qwen3-ASR for DetectLanguage.");
    }

    /// <summary>Releases CPU model weights and graph workspaces. Repeated calls are harmless.</summary>
    public void Dispose()
    {
        if (disposed) return; disposed = true;
        encoderFirst.Dispose(); encoderSecond.Dispose(); predictor.Dispose(); joint.Dispose();
    }
}
