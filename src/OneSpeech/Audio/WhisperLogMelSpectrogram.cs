using System.Numerics;

namespace OneSpeech.Audio;

/// <summary>Whisper's 16 kHz mono log-mel front end, with 400-sample periodic Hann windows and 160-sample hops.</summary>
/// <remarks>Instances are immutable and thread-safe. No resampling or downmixing is performed.</remarks>
public sealed class WhisperLogMelSpectrogram
{
    /// <summary>Required input sample rate in Hz.</summary>
    public const int SampleRate = 16000;
    /// <summary>Maximum samples per model window (30 seconds).</summary>
    public const int WindowSamples = 480000;
    /// <summary>Frames in a zero-padded 30-second window.</summary>
    public const int WindowFrames = 3000;
    private const int FftSize = 400;
    private const int Hop = 160;
    private const int Frequencies = 201;
    private readonly float[] hann = new float[FftSize];
    private readonly float[] filters;
    private readonly Complex[] roots = new Complex[FftSize];

    /// <summary>Creates an extractor with 80 bins (tiny through large-v2) or 128 (large-v3 and turbo).</summary>
    /// <exception cref="ArgumentOutOfRangeException">The bin count is neither 80 nor 128.</exception>
    public WhisperLogMelSpectrogram(int melBins = 80)
    {
        if (melBins is not (80 or 128)) throw new ArgumentOutOfRangeException(nameof(melBins));
        MelBins = melBins;
        for (int i = 0; i < FftSize; i++)
        {
            hann[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / FftSize));
            roots[i] = Complex.FromPolarCoordinates(1, -2 * Math.PI * i / FftSize);
        }
        filters = CreateFilters(melBins);
    }

    /// <summary>Number of mel channels; 80 by default.</summary>
    public int MelBins { get; }

    /// <summary>Extracts mel-major float features. With padding (default), returns <c>MelBins * 3000</c> values.</summary>
    /// <remarks>Short input is zero-padded before reflection at the window edges. Without padding, frame count is sample count / 160;
    /// inputs of 200 samples or fewer are rejected because centered reflection requires more than half an FFT window.
    /// The last STFT frame is dropped. Power is floored at 1e-10, log10-clipped to peak minus 8, then mapped by (x + 4) / 4.
    /// Input samples must be finite; they are not clipped. Cancellation is checked per frame. Storage is bounded by one 30-second window.</remarks>
    /// <exception cref="ArgumentNullException">Audio is null.</exception>
    /// <exception cref="ArgumentException">Input is not 16 kHz mono, exceeds 30 seconds, contains nonfinite samples, exceeds float32 power range, or is too short without padding.</exception>
    /// <exception cref="OperationCanceledException">Cancellation is requested.</exception>
    public float[] Extract(AudioBuffer audio, bool padToWindow = true, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.SampleRate != SampleRate || audio.Channels != 1 || audio.FrameCount > WindowSamples)
            throw new ArgumentException("Expected at most 30 seconds of 16 kHz mono audio.", nameof(audio));
        ReadOnlySpan<float> samples = audio.Samples.Span;
        foreach (float sample in samples)
            if (!float.IsFinite(sample)) throw new ArgumentException("Audio samples must be finite.", nameof(audio));
        int length = padToWindow ? WindowSamples : samples.Length;
        if (length <= FftSize / 2) throw new ArgumentException("Unpadded centered STFT requires more than 200 samples.", nameof(audio));
        cancellationToken.ThrowIfCancellationRequested();
        return Compute(audio, length, 0, length / Hop, cancellationToken);
    }

    internal float[] ExtractQwen(AudioBuffer audio, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.SampleRate != SampleRate || audio.Channels != 1 || audio.FrameCount is <= 200 or > 4800000)
            throw new ArgumentException("Qwen features require more than 200 samples and at most 300 seconds of 16 kHz mono audio.", nameof(audio));
        foreach (float sample in audio.Samples.Span)
            if (!float.IsFinite(sample)) throw new ArgumentException("Audio samples must be finite.", nameof(audio));
        return Compute(audio, audio.FrameCount, 0, audio.FrameCount / Hop, cancellationToken);
    }

    internal float[] ExtractContext(AudioBuffer audio, int offset, float floor, CancellationToken cancellationToken)
    {
        if (offset < 0 || offset >= audio.FrameCount) throw new ArgumentOutOfRangeException(nameof(offset));
        float[] result = Compute(audio, (long)audio.FrameCount + WindowSamples, offset, WindowFrames, cancellationToken);
        for (int i = 0; i < result.Length; i++) result[i] = MathF.Max(result[i], floor);
        return result;
    }

    private float[] Compute(AudioBuffer audio, long length, int offset, int frames, CancellationToken cancellationToken)
    {
        ReadOnlySpan<float> samples = audio.Samples.Span;
        float[] features = new float[MelBins * frames];
        Complex[] input = new Complex[FftSize];
        Complex[] output = new Complex[FftSize];
        Complex[] scratch = new Complex[FftSize];
        float[] power = new float[Frequencies];
        float peak = float.NegativeInfinity;
        // Radford et al. 2022 §2.2: 25 ms windows and 10 ms stride at 16 kHz.
        for (int frame = 0; frame < frames; frame++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < FftSize; i++)
            {
                long index = (long)offset + frame * Hop + i - FftSize / 2;
                if (index < 0) index = -index;
                else if (index >= length) index = 2 * length - 2 - index;
                input[i] = index < samples.Length ? samples[(int)index] * hann[i] : 0;
            }
            Transform(input, 0, 1, output, scratch);
            for (int i = 0; i < Frequencies; i++)
            {
                float real = (float)output[i].Real;
                float imaginary = (float)output[i].Imaginary;
                power[i] = real * real + imaginary * imaginary;
                if (!float.IsFinite(power[i])) throw new ArgumentException("Audio power exceeds float32 range.", nameof(audio));
            }
            for (int mel = 0; mel < MelBins; mel++)
            {
                float energy = 0;
                for (int frequency = 0; frequency < Frequencies; frequency++)
                    energy += filters[mel * Frequencies + frequency] * power[frequency];
                if (!float.IsFinite(energy)) throw new ArgumentException("Mel energy exceeds float32 range.", nameof(audio));
                float value = MathF.Log10(MathF.Max(energy, 1e-10f));
                features[mel * frames + frame] = value;
                peak = MathF.Max(peak, value);
            }
        }
        for (int i = 0; i < features.Length; i++) features[i] = (MathF.Max(features[i], peak - 8) + 4) / 4;
        return features;
    }

    // Mixed-radix Cooley–Tukey: all lengths divide 400; recursion depth is six.
    // The three 400-element work arrays are allocated once per extraction, never per frame.
    private void Transform(Complex[] input, int start, int stride, Span<Complex> output, Span<Complex> scratch)
    {
        int length = output.Length;
        if (length == 1) { output[0] = input[start]; return; }
        int radix = length % 2 == 0 ? 2 : 5;
        int subLength = length / radix;
        for (int r = 0; r < radix; r++)
            Transform(input, start + r * stride, stride * radix, output.Slice(r * subLength, subLength), scratch.Slice(r * subLength, subLength));
        for (int k = 0; k < length; k++)
        {
            Complex sum = Complex.Zero;
            for (int r = 0; r < radix; r++) sum += output[r * subLength + k % subLength] * roots[r * k * (FftSize / length) % FftSize];
            scratch[k] = sum;
        }
        scratch.CopyTo(output);
    }

    private static float[] CreateFilters(int count)
    {
        // Slaney's piecewise mel scale, triangular filters, normalized by their Hz bandwidth.
        double[] edges = new double[count + 2];
        double maxMel = HzToMel(SampleRate / 2);
        for (int i = 0; i < edges.Length; i++) edges[i] = MelToHz(maxMel * i / (count + 1));
        float[] result = new float[count * Frequencies];
        for (int m = 0; m < count; m++)
            for (int f = 0; f < Frequencies; f++)
            {
                double hz = (double)SampleRate * f / FftSize;
                double triangle = Math.Max(0, Math.Min((hz - edges[m]) / (edges[m + 1] - edges[m]), (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
                // Round the unnormalized triangle before applying the area normalization.
                result[m * Frequencies + f] = (float)((float)triangle * (2 / (edges[m + 2] - edges[m])));
            }
        return result;
    }

    private static double HzToMel(double hz) => hz < 1000 ? hz / (200.0 / 3) : 15 + Math.Log(hz / 1000) / (Math.Log(6.4) / 27);
    private static double MelToHz(double mel) => mel < 15 ? mel * (200.0 / 3) : 1000 * Math.Exp((mel - 15) * (Math.Log(6.4) / 27));
}
