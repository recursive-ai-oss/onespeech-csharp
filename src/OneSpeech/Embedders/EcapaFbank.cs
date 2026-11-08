namespace OneSpeech.Embedders;

/// <summary>ECAPA's 80-band log-mel profile for mono 16,000 Hz float PCM, verified against the pinned
/// SpeechBrain reference: 25 ms periodic Hamming window, 10 ms hop, centered zero padding and 400-point FFT.
/// This profile is not interchangeable with the CAM++ speaker filterbank.</summary>
public sealed class EcapaFbank
{
    private const int Window = 400, Hop = 160, Bins = 201, Mels = 80;
    private readonly float[] window = new float[Window];
    private readonly float[] filters = new float[Mels * Bins];
    private readonly double[] cosine = new double[Bins * 25], sine = new double[Bins * 25];
    private readonly double[] cosine16 = new double[8], sine16 = new double[8];

    /// <summary>Creates fixed coefficients. No randomization, preemphasis, DC removal or amplitude scaling is used.</summary>
    public EcapaFbank()
    {
        float[] points = new float[82];
        float melMax = 2595 * MathF.Log10(1 + 8000f / 700);
        for (int i = 0; i < points.Length; i++) points[i] = 700 * (MathF.Pow(10, i * (melMax / 81) / 2595) - 1);
        for (int j = 0; j < Window; j++) window[j] = .54f - .46f * MathF.Cos(2 * MathF.PI * j / Window);
        // Public numeric probes distinguish symmetric Hz triangles using the previous center spacing
        // from conventional asymmetric adjacent-point triangles.
        for (int m = 0; m < Mels; m++) for (int b = 0; b < Bins; b++)
            filters[m * Bins + b] = Math.Max(0, 1 - Math.Abs(40 * b - points[m + 1]) / (points[m + 1] - points[m]));
        for (int b = 0; b < Bins; b++) for (int j = 0; j < 25; j++)
        {
            double angle = -2 * Math.PI * b * j / Window;
            cosine[b * 25 + j] = Math.Cos(angle); sine[b * 25 + j] = Math.Sin(angle);
        }
        for (int i = 0; i < 8; i++) { cosine16[i] = Math.Cos(-2 * Math.PI * i / 16); sine16[i] = Math.Sin(-2 * Math.PI * i / 16); }
    }

    /// <summary>Returns centered-frame count (floor(samples/160)+1). Accepts 1 through 960,000 samples
    /// (up to 60 s at 16,000 Hz); throws ArgumentOutOfRangeException otherwise.</summary>
    public static int FrameCount(int sampleCount)
    {
        if (sampleCount is < 1 or > 960000) throw new ArgumentOutOfRangeException(nameof(sampleCount), "ECAPA features require nonempty PCM of at most 60 s at 16000 Hz.");
        return sampleCount / Hop + 1;
    }

    /// <summary>Returns mean-normalized [mel, frame] row-major features. Uses the same limits as ComputeRaw;
    /// checks cancellation each frame. No samples or features are retained.</summary>
    public float[] Compute(ReadOnlySpan<float> samples, CancellationToken cancellationToken)
    {
        float[] output = ComputeRaw(samples, cancellationToken);
        int frames = output.Length / Mels;
        for (int m = 0; m < Mels; m++)
        {
            float mean = 0;
            for (int f = 0; f < frames; f++) mean += output[m * frames + f];
            mean /= frames;
            for (int f = 0; f < frames; f++) output[m * frames + f] -= mean;
        }
        return output;
    }

    /// <summary>Returns [mel, frame] decibel energies before mean subtraction. Power is floored at 1e-10;
    /// values below the utterance maximum minus 80 dB are clipped. See FrameCount for limits.
    /// Throws ArgumentException for nonfinite samples or magnitudes that overflow feature computation;
    /// checks cancellation each frame. Returns newly allocated storage.</summary>
    public float[] ComputeRaw(ReadOnlySpan<float> samples, CancellationToken cancellationToken)
    {
        int frames = FrameCount(samples.Length);
        foreach (float value in samples) if (!float.IsFinite(value)) throw new ArgumentException("ECAPA PCM must be finite.", nameof(samples));
        float[] output = new float[Mels * frames], input = new float[Window], power = new float[Bins];
        double[] re16 = new double[16], im16 = new double[16], stageRe = new double[Window], stageIm = new double[Window];
        float maximum = float.NegativeInfinity;
        for (int f = 0; f < frames; f++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int j = 0; j < Window; j++)
            {
                int index = f * Hop + j - Window / 2;
                input[j] = index >= 0 && index < samples.Length ? samples[index] * window[j] : 0;
            }
            Power(input, power, re16, im16, stageRe, stageIm);
            for (int m = 0; m < Mels; m++)
            {
                float sum = 0;
                for (int b = 0; b < Bins; b++) sum += filters[m * Bins + b] * power[b];
                if (!float.IsFinite(sum)) throw new ArgumentException("PCM magnitude overflows ECAPA feature computation.", nameof(samples));
                float value = 10 * MathF.Log10(Math.Max(sum, 1e-10f));
                output[m * frames + f] = value; maximum = Math.Max(maximum, value);
            }
        }
        for (int i = 0; i < output.Length; i++) output[i] = Math.Max(output[i], maximum - 80);
        return output;
    }

    // Fixed 400=25*16 mixed-radix transform. All work arrays have the stated fixed sizes;
    // every stage index is <400, every final coefficient index <201*25, and twiddle indices <8.
    private void Power(float[] input, float[] output, double[] real, double[] imaginary, double[] stageRe, double[] stageIm)
    {
        for (int group = 0; group < 25; group++)
        {
            for (int i = 0; i < 16; i++) { real[i] = input[25 * i + group]; imaginary[i] = 0; }
            for (int i = 1, j = 0; i < 16; i++)
            {
                int bit = 8;
                for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) { (real[i], real[j]) = (real[j], real[i]); (imaginary[i], imaginary[j]) = (imaginary[j], imaginary[i]); }
            }
            for (int length = 2; length <= 16; length <<= 1)
            {
                int half = length / 2;
                for (int k = 0; k < half; k++)
                {
                    double wr = cosine16[k * (16 / length)], wi = sine16[k * (16 / length)];
                    for (int start = 0; start < 16; start += length)
                    {
                        int a = start + k, b = a + half;
                        double tr = wr * real[b] - wi * imaginary[b], ti = wr * imaginary[b] + wi * real[b];
                        real[b] = real[a] - tr; imaginary[b] = imaginary[a] - ti; real[a] += tr; imaginary[a] += ti;
                    }
                }
            }
            for (int k = 0; k < 16; k++) { stageRe[group * 16 + k] = real[k]; stageIm[group * 16 + k] = imaginary[k]; }
        }
        for (int b = 0; b < Bins; b++)
        {
            double re = 0, im = 0;
            for (int group = 0; group < 25; group++)
            {
                int index = group * 16 + b % 16, coefficient = b * 25 + group;
                re += stageRe[index] * cosine[coefficient] - stageIm[index] * sine[coefficient];
                im += stageRe[index] * sine[coefficient] + stageIm[index] * cosine[coefficient];
            }
            output[b] = (float)(re * re + im * im);
        }
    }
}
