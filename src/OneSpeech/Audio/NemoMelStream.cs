using System.Numerics;

namespace OneSpeech.Audio;

// Session-local, allocation-free raw NeMo mel features. The emitted array is borrowed only
// for the callback; the consumer must copy or consume it before the next frame is emitted.
internal sealed class NemoMelStream(NemoLogMelSpectrogram extractor)
{
    private readonly float[] ring = new float[1024];
    private readonly float[] emphasized = new float[512];
    private readonly float[] powers = new float[257];
    private readonly Complex[] fft = new Complex[512];
    private readonly float[] mel = new float[extractor.MelBins];
    private long samples;
    private long frame;
    private float previous;
    private bool completed;

    internal long FrameCount => frame;

    internal void Push(ReadOnlySpan<float> audio, bool final, Action<float[]> emit, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(emit);
        if (completed) throw new InvalidOperationException("Feature stream is complete; reset before reuse.");
        cancellationToken.ThrowIfCancellationRequested();
        foreach (float sample in audio)
        {
            if (!float.IsFinite(sample)) throw new ArgumentException("PCM must be finite.", nameof(audio));
            float value = samples == 0 ? sample : sample - .97f * previous;
            previous = sample; ring[(int)(samples % ring.Length)] = value; samples++;
            // Only the 400-sample Hann support needs future PCM; the 512-point FFT's
            // outer zero padding does not add another 56 samples of input latency.
            if (samples >= frame * 160 + 200)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Emit(emit); frame++;
            }
        }
        if (!final) return;
        while (frame < samples / 160)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Emit(emit); frame++;
        }
        completed = true;
    }

    private void Emit(Action<float[]> emit)
    {
        long first = frame * 160 - 256;
        for (int i = 0; i < emphasized.Length; i++)
        {
            long at = first + i;
            emphasized[i] = at < 0 || at >= samples ? 0 : ring[(int)(at % ring.Length)];
        }
        extractor.ExtractUnnormalizedFrame(emphasized, mel, fft, powers);
        emit(mel);
    }

    internal void Reset()
    {
        Array.Clear(ring); Array.Clear(emphasized); Array.Clear(mel); Array.Clear(powers); Array.Clear(fft);
        samples = frame = 0; previous = 0; completed = false;
    }
}
