using OneAudio;

namespace OneSpeech.Audio;

/// <summary>Bounded pull decoding to 16 kHz interleaved float PCM. Original channels remain separate; recognition mixes them later. Not thread-safe.</summary>
public sealed class AudioFileSource : IDisposable
{
    private readonly Stream stream;
    private readonly bool leaveOpen;
    private readonly IAudioDecoder decoder;
    private readonly IResampler? resampler;
    private readonly float[] input;
    private int available, offset;
    private bool ended;

    private AudioFileSource(Stream stream, bool leaveOpen, CancellationToken cancellationToken)
    {
        this.stream = stream;
        this.leaveOpen = leaveOpen;
        long origin = stream.Position;
        var g711 = G711WaveDecoder.TryOpen(stream);
        if (g711 is null) stream.Position = origin;
        decoder = g711 ?? OneAudio.Audio.Open(stream, leaveOpen: true, cancellationToken: cancellationToken);
        Channels = decoder.Format.Channels;
        int rate = decoder.Format.SampleRate;
        if (rate is < 250 or > 768000 || Channels is < 1 or > 32)
        {
            decoder.Dispose();
            throw new UnsupportedAudioException("Audio rate must be 250–768000 Hz with 1–32 channels.");
        }
        input = new float[2048 * Channels];
        if (rate != 16000)
        {
            double ratio = 16000d / rate;
            resampler = AudioResampler.Create(rate, 16000, Channels, ResamplerQuality.Polyphase,
                new ResamplerOptions { MinRatio = ratio, MaxRatio = ratio });
        }
    }

    /// <summary>Original channel count. All output blocks preserve this interleaving.</summary>
    public int Channels { get; }
    /// <summary>Output sample rate in Hz, always 16000.</summary>
    public const int OutputSampleRate = 16000;

    /// <summary>Opens a file for bounded decoding. Throws UnsupportedAudioException for corrupt/unsupported audio; filesystem errors remain IOException. Checks cancellation during open.</summary>
    public static AudioFileSource Open(string path, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        cancellationToken.ThrowIfCancellationRequested();
        var file = File.OpenRead(path);
        try { return Open(file, false, cancellationToken); }
        catch { file.Dispose(); throw; }
    }

    /// <summary>Opens a readable seekable stream at its current position; ownership follows leaveOpen. Throws ArgumentException for nonseekable streams; UnsupportedAudioException for invalid audio.</summary>
    public static AudioFileSource Open(Stream stream, bool leaveOpen, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead || !stream.CanSeek) throw new ArgumentException("Audio decoding requires a readable seekable stream. Use the asynchronous pipeline for nonseekable input.", nameof(stream));
        cancellationToken.ThrowIfCancellationRequested();
        try { return new AudioFileSource(stream, leaveOpen, cancellationToken); }
        catch (Exception ex) when (ex is OneAudioException or EndOfStreamException or NotSupportedException)
        {
            if (!leaveOpen) stream.Dispose();
            throw new UnsupportedAudioException("Unsupported or corrupt audio. Supply a supported OneAudio file or a G.711 WAV.", ex);
        }
        catch { if (!leaveOpen) stream.Dispose(); throw; }
    }

    /// <summary>Reads at most destination.Length/Channels frames; returns zero only at EOF or for a sub-frame destination. Cancellation is checked per decoder/resampler block. Throws UnsupportedAudioException for corrupt payloads.</summary>
    public int Read(Span<float> destination, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (destination.Length < Channels) return 0;
        try
        {
            if (resampler is null) return decoder.Read(destination);
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (offset == available && !ended)
                {
                    available = decoder.Read(input) * Channels;
                    offset = 0;
                    ended = available == 0;
                }
                if (ended) return resampler.Flush(destination);
                resampler.Process(input.AsSpan(offset, available - offset), destination, out int consumed, out int written);
                offset += consumed * Channels;
                if (written != 0) return written;
            }
        }
        catch (Exception ex) when (ex is OneAudioException or EndOfStreamException)
        {
            throw new UnsupportedAudioException("The audio payload is corrupt or truncated.", ex);
        }
    }

    /// <summary>Disposes the decoder and, unless leaveOpen was requested, the input stream.</summary>
    public void Dispose()
    {
        decoder.Dispose();
        if (!leaveOpen) stream.Dispose();
    }
}
