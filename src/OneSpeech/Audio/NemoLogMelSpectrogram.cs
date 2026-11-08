using System.Numerics;

namespace OneSpeech.Audio;

/// <summary>Deterministic 16 kHz NeMo power log-mel front end: 25 ms Hann window, 10 ms hop, 512-point FFT and Slaney filters.</summary>
/// <remarks>No resampling, clipping, noise dithering or augmentation is performed. Instances are immutable and thread-safe.</remarks>
public sealed class NemoLogMelSpectrogram
{
    /// <summary>Required mono PCM sample rate, in Hz.</summary>
    public const int SampleRate = 16000;
    /// <summary>Maximum input samples, corresponding to 60 seconds. Longer audio must be chunked.</summary>
    public const int MaximumSamples = 960000;
    private const int FftSize = 512;
    private const int Frequencies = 257;
    private const int Hop = 160;
    private readonly float[] window = new float[FftSize];
    private readonly float[] filters;
    private readonly Complex[] roots = new Complex[FftSize / 2];

    /// <summary>Creates an extractor. The default 128 bins match Parakeet v3; 80-bin checkpoints are also supported.</summary>
    /// <exception cref="ArgumentOutOfRangeException">melBins is neither 80 nor 128.</exception>
    public NemoLogMelSpectrogram(int melBins = 128)
    {
        if (melBins is not (80 or 128)) throw new ArgumentOutOfRangeException(nameof(melBins));
        MelBins = melBins;
        for (int i = 0; i < 400; i++) window[i + 56] = (float)(.5 - .5 * Math.Cos(2 * Math.PI * i / 399));
        for (int i = 0; i < roots.Length; i++) roots[i] = Complex.FromPolarCoordinates(1, -2 * Math.PI * i / FftSize);
        filters = CreateFilters(melBins);
    }

    /// <summary>Mel channels, 128 by default.</summary>
    public int MelBins { get; }

    /// <summary>Returns the valid centered STFT frame count, before optional time padding.</summary>
    /// <param name="sampleCount">PCM length, range 320..960000.</param>
    /// <exception cref="ArgumentOutOfRangeException">The length is outside the supported range.</exception>
    public static int GetFrameCount(int sampleCount)
    {
        if (sampleCount is < 320 or > MaximumSamples) throw new ArgumentOutOfRangeException(nameof(sampleCount));
        return sampleCount / Hop;
    }

    /// <summary>Extracts mel-major features with preemphasis 0.97, centered zero padding, natural log with additive 2^-24 guard,
    /// and per-channel sample-standard-deviation normalization (epsilon 1e-5). Time padding is zero, after normalization.</summary>
    /// <param name="audio">Finite 16 kHz mono PCM, 320..960000 samples. No resampling/downmixing is done.</param>
    /// <param name="padTo">Time-axis multiple; default zero (no rounding). Range 0..1024. The centered STFT adds one masked trailing frame.</param>
    /// <param name="normalize">Per-feature normalization, true by default; false returns natural-log energies.</param>
    /// <param name="cancellationToken">Checked per STFT frame and normalization channel.</param>
    /// <exception cref="ArgumentException">PCM shape, length, values or padding are invalid, or power exceeds float32 range.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public float[] Extract(AudioBuffer audio, int padTo = 0, bool normalize = true, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.SampleRate != SampleRate || audio.Channels != 1 || padTo is < 0 or > 1024) throw new ArgumentException("Expected 16 kHz mono and padding in 0..1024.");
        int frames = GetFrameCount(audio.FrameCount), padded = padTo == 0 ? frames + 1 : (frames + padTo) / padTo * padTo;
        var samples = audio.Samples.Span;
        float[] emphasized = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            if (!float.IsFinite(samples[i])) throw new ArgumentException("PCM must be finite.", nameof(audio));
            emphasized[i] = i == 0 ? samples[i] : samples[i] - .97f * samples[i - 1];
        }
        float[] result = new float[MelBins * padded], power = new float[Frequencies];
        Complex[] fft = new Complex[FftSize];
        for (int frame = 0; frame < frames; frame++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < FftSize; i++)
            {
                int index = frame * Hop + i - FftSize / 2;
                fft[i] = (uint)index < emphasized.Length ? emphasized[index] * window[i] : 0;
            }
            Transform(fft);
            for (int f = 0; f < Frequencies; f++)
            {
                float real = (float)fft[f].Real, imaginary = (float)fft[f].Imaginary;
                power[f] = real * real + imaginary * imaginary;
                if (!float.IsFinite(power[f])) throw new ArgumentException("Power exceeds float32 range.", nameof(audio));
            }
            for (int mel = 0; mel < MelBins; mel++)
            {
                float energy = 0;
                for (int f = 0; f < Frequencies; f++) energy += filters[mel * Frequencies + f] * power[f];
                if (!float.IsFinite(energy)) throw new ArgumentException("Mel energy exceeds float32 range.", nameof(audio));
                result[mel * padded + frame] = MathF.Log(energy + 1f / 16777216);
            }
        }
        if (normalize)
            for (int mel = 0; mel < MelBins; mel++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var channel = result.AsSpan(mel * padded, frames);
                // Float32 mean arithmetic retains the low-variance behavior observed in the reference output.
                // Four independent accumulators avoid a length-dependent serial-sum error on ordinary speech.
                float a = 0, b = 0, c = 0, d = 0;
                int index = 0;
                for (; index + 3 < frames; index += 4) { a += channel[index]; b += channel[index + 1]; c += channel[index + 2]; d += channel[index + 3]; }
                for (; index < frames; index++) a += channel[index];
                float mean = ((a + b) + c + d) / frames;
                double variance = 0;
                foreach (float value in channel) variance += (value - mean) * (value - mean);
                float scale = MathF.Sqrt((float)(variance / (frames - 1))) + 1e-5f;
                for (int i = 0; i < frames; i++) channel[i] = (float)((channel[i] - mean) / scale);
            }
        return result;
    }

    private void Transform(Complex[] values)
    {
        for (int i = 1, j = 0; i < FftSize; i++)
        {
            int bit = FftSize >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (values[i], values[j]) = (values[j], values[i]);
        }
        for (int length = 2; length <= FftSize; length *= 2)
            for (int start = 0; start < FftSize; start += length)
                for (int j = 0; j < length / 2; j++)
                {
                    var even = values[start + j];
                    var odd = roots[j * (FftSize / length)] * values[start + j + length / 2];
                    values[start + j] = even + odd; values[start + j + length / 2] = even - odd;
                }
    }

    private static float[] CreateFilters(int count)
    {
        // Slaney's piecewise mel scale and area normalization, as declared by the preprocessor API.
        static double Mel(double hz) => hz < 1000 ? hz / (200.0 / 3) : 15 + Math.Log(hz / 1000) / (Math.Log(6.4) / 27);
        static double Hz(double mel) => mel < 15 ? mel * (200.0 / 3) : 1000 * Math.Exp((mel - 15) * Math.Log(6.4) / 27);
        double[] edges = new double[count + 2];
        for (int i = 0; i < edges.Length; i++) edges[i] = Hz(Mel(8000) * i / (count + 1));
        float[] weights = new float[count * Frequencies];
        for (int m = 0; m < count; m++)
            for (int f = 0; f < Frequencies; f++)
            {
                double hz = (double)SampleRate * f / FftSize;
                float triangle = (float)Math.Max(0, Math.Min((hz - edges[m]) / (edges[m + 1] - edges[m]), (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
                weights[m * Frequencies + f] = (float)(triangle * (2 / (edges[m + 2] - edges[m])));
            }
        return weights;
    }
}
