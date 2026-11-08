namespace OneSpeech.Tests;

// Unoptimized scalar baseline kept independent of the sparse-filter/twiddle-cache
// implementation so arithmetic-order regressions have an exact comparator.
/// <summary>80-band speaker log-filterbanks for mono 16,000 Hz PCM: 25 ms windows, 10 ms hop,
/// no dithering, and per-recording mean subtraction. Not interchangeable with Whisper features.</summary>
internal sealed class DenseSpeakerFbank
{
    private const int Window = 400, Hop = 160, FftSize = 512, Bins = 257, Mels = 80;
    private readonly float[] window = new float[Window];
    private readonly float[] filters = new float[Mels * Bins];

    /// <summary>Creates fixed filter and window coefficients. There are no random or trainable parameters.</summary>
    public DenseSpeakerFbank()
    {
        for (int i = 0; i < Window; i++) window[i] = MathF.Pow(.5f - .5f * MathF.Cos(2 * MathF.PI * i / (Window - 1)), .85f);
        float low = Mel(20), high = Mel(8000), step = (high - low) / (Mels + 1);
        for (int m = 0; m < Mels; m++)
        {
            float left = low + m * step, middle = low + (m + 1) * step, right = low + (m + 2) * step;
            for (int k = 0; k < Bins; k++)
            {
                float f = Mel(k * (16000f / FftSize));
                filters[m * Bins + k] = Math.Max(0, Math.Min((f - left) / (middle - left), (right - f) / (right - middle)));
            }
        }
    }

    /// <summary>Returns complete snipped-frame count. Requires 400 through 960,000 samples (25 ms through 60 s);
    /// other counts throw ArgumentOutOfRangeException. No partial final window is padded.</summary>
    public static int FrameCount(int sampleCount)
    {
        if (sampleCount is < Window or > 960000) throw new ArgumentOutOfRangeException(nameof(sampleCount), "Speaker features require 25 ms through 60 s at 16000 Hz.");
        return 1 + (sampleCount - Window) / Hop;
    }

    /// <summary>Computes [mel, frame] row-major mean-normalized natural-log energies. Samples must be finite;
    /// nonfinite samples or finite magnitudes that overflow feature computation throw ArgumentException. See FrameCount for length limits. Checks cancellation each frame.
    /// Float PCM is used without amplitude scaling, followed by DC removal and 0.97 preemphasis. Energy is floored at float machine epsilon.</summary>
    public float[] Compute(ReadOnlySpan<float> samples, CancellationToken cancellationToken)
    {
        int frames = FrameCount(samples.Length);
        foreach (float value in samples) if (!float.IsFinite(value)) throw new ArgumentException("Speaker PCM must be finite.", nameof(samples));
        float[] result = new float[Mels * frames], real = new float[FftSize], imaginary = new float[FftSize], power = new float[Bins];
        for (int frame = 0; frame < frames; frame++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            real.AsSpan().Clear(); imaginary.AsSpan().Clear();
            int start = frame * Hop;
            float mean = 0;
            for (int j = 0; j < Window; j++) { real[j] = samples[start + j]; mean += real[j]; }
            mean /= Window;
            for (int j = 0; j < Window; j++) real[j] -= mean;
            for (int j = Window - 1; j > 0; j--) real[j] = (real[j] - .97f * real[j - 1]) * window[j];
            real[0] = (real[0] - .97f * real[0]) * window[0];
            Transform(real, imaginary);
            for (int k = 0; k < Bins; k++) power[k] = real[k] * real[k] + imaginary[k] * imaginary[k];
            for (int m = 0; m < Mels; m++)
            {
                float sum = 0;
                for (int k = 0; k < Bins; k++) sum += power[k] * filters[m * Bins + k];
                if (!float.IsFinite(sum)) throw new ArgumentException("PCM magnitude overflows speaker feature computation.", nameof(samples));
                result[m * frames + frame] = MathF.Log(Math.Max(1.1920928955078125e-7f, sum));
            }
        }
        for (int m = 0; m < Mels; m++)
        {
            float mean = 0;
            for (int f = 0; f < frames; f++) mean += result[m * frames + f];
            mean /= frames;
            for (int f = 0; f < frames; f++) result[m * frames + f] -= mean;
        }
        return result;
    }

    private static float Mel(float frequency) => 1127 * MathF.Log(1 + frequency / 700);
    // Fixed 512-element radix-2 FFT: every index is within the two equally sized arrays.
    private static void Transform(float[] real, float[] imaginary)
    {
        for (int i = 1, j = 0; i < FftSize; i++)
        {
            int bit = FftSize >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { (real[i], real[j]) = (real[j], real[i]); (imaginary[i], imaginary[j]) = (imaginary[j], imaginary[i]); }
        }
        for (int length = 2; length <= FftSize; length <<= 1)
        {
            int half = length / 2;
            for (int k = 0; k < half; k++)
            {
                float angle = -2 * MathF.PI * k / length, wr = MathF.Cos(angle), wi = MathF.Sin(angle);
                for (int start = 0; start < FftSize; start += length)
                {
                    int a = start + k, b = a + half;
                    float tr = wr * real[b] - wi * imaginary[b], ti = wr * imaginary[b] + wi * real[b];
                    real[b] = real[a] - tr; imaginary[b] = imaginary[a] - ti;
                    real[a] += tr; imaginary[a] += ti;
                }
            }
        }
    }
}
