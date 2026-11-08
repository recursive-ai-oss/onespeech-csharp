using OneSpeech.Audio;

namespace OneSpeech.Diarization;

// Holds only a 30.4 s window plus STFT edge context. No waveform or mel sequence grows with a file.
internal static class NemotronFeatureChunks
{
    internal sealed record Chunk(float[] Stacked, int Frames, int ValidFrames, int OutputMelFrames, long SampleEnd);
    internal static IEnumerable<Chunk> Read(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken)
    {
        using var reader = new PcmReader(blocks.GetEnumerator());
        const int pastSamples = 480, chunkSamples = 340 * 1280, windowSamples = 380 * 1280;
        float[] buffer = new float[pastSamples + windowSamples + 512];
        int count = 0, past = 0;
        bool eof = false;
        var extractor = new NemoLogMelSpectrogram();
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!eof)
            {
                int wanted = past + windowSamples + 512 - count;
                int read = reader.Read(buffer.AsSpan(count, wanted), cancellationToken);
                count += read; eof = read < wanted;
            }
            int samples = count - past;
            if (reader.Total == 0) yield break;
            int validMel = eof ? Math.Min(3040, samples / 160 + 1) : 3040;
            int frames = (validMel + 7) / 8;
            int validCenteredMel = eof ? Math.Min(3040, samples / 160) : validMel;
            int validFrames = (validCenteredMel + 7) / 8;
            // The final centered STFT frame uses zero right padding, including sub-hop tails.
            float[] pcm = new float[Math.Max(320, count)];
            buffer.AsSpan(0, count).CopyTo(pcm);
            float[] mel = extractor.Extract(new AudioBuffer(pcm, 16000, 1), normalize: false, cancellationToken: cancellationToken);
            int stride = pcm.Length / 160 + 1;
            float[] stacked = new float[frames * 1024];
            for (int t = 0; t < validMel; t++)
                // The reference masks its centered trailing frame to zero, but retains its position.
                for (int m = 0; m < 128; m++) stacked[t * 128 + m] = eof && t == samples / 160 ? 0 : mel[m * stride + t + past / 160];
            yield return new Chunk(stacked, frames, validFrames, Math.Min(2720, validMel), reader.Total);
            if (eof && validMel <= 2720) yield break;
            int remove = past + chunkSamples - pastSamples;
            buffer.AsSpan(remove, count - remove).CopyTo(buffer);
            count -= remove; past = pastSamples;
        }
    }

    private sealed class PcmReader(IEnumerator<AudioBuffer> blocks) : IDisposable
    {
        private AudioBuffer? current;
        private int position;
        internal long Total { get; private set; }
        internal int Read(Span<float> destination, CancellationToken cancellationToken)
        {
            int written = 0;
            while (written < destination.Length)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (current is null || position == current.FrameCount)
                {
                    if (!blocks.MoveNext()) break;
                    current = blocks.Current ?? throw new ArgumentException("Null PCM block.");
                    position = 0;
                    if (current.SampleRate != 16000) throw new ArgumentException("Nemotron diarization requires 16 kHz PCM. Resample with OneAudio.");
                    if (current.FrameCount == 0) continue;
                }
                int count = Math.Min(destination.Length - written, current.FrameCount - position);
                var samples = current.Samples.Span;
                for (int i = 0; i < count; i++)
                {
                    double sum = 0;
                    for (int ch = 0; ch < current.Channels; ch++)
                    {
                        float value = samples[(position + i) * current.Channels + ch];
                        if (!float.IsFinite(value)) throw new ArgumentException("PCM must be finite.");
                        sum += value;
                    }
                    destination[written + i] = (float)(sum / current.Channels);
                }
                position += count; written += count; Total += count;
            }
            return written;
        }
        public void Dispose() => blocks.Dispose();
    }
}
