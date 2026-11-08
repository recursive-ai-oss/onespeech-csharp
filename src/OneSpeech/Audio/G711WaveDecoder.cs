using System.Buffers.Binary;
using OneAudio;

namespace OneSpeech.Audio;

// Temporary adapter until OneAudio supports WAVE format tags 6 and 7.
internal sealed class G711WaveDecoder : IAudioDecoder
{
    private readonly Stream stream;
    private readonly long dataStart;
    private readonly bool muLaw;
    private readonly byte[] bytes = new byte[8192];
    private bool disposed;

    private G711WaveDecoder(Stream stream, int rate, int channels, long start, long length, bool muLaw)
    {
        this.stream = stream;
        dataStart = start;
        LengthFrames = length / channels;
        Format = new AudioFormat(rate, channels, ChannelMask.None, NativeSampleType.Lossy);
        this.muLaw = muLaw;
    }

    internal static IAudioDecoder? TryOpen(Stream stream)
    {
        if (!stream.CanSeek) throw new ArgumentException("G.711 probing needs a seekable stream.", nameof(stream));
        long origin = stream.Position;
        try
        {
            var header = new byte[12];
            if (stream.ReadAtLeast(header, header.Length, throwOnEndOfStream: false) < header.Length || !header.AsSpan(0, 4).SequenceEqual("RIFF"u8) || !header.AsSpan(8, 4).SequenceEqual("WAVE"u8)) return null;
            long end = checked(origin + 8 + BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(4)));
            if (end > stream.Length || end < stream.Position) throw Invalid("Truncated RIFF extent.");
            var chunk = new byte[8];
            int tag = 0, channels = 0, rate = 0;
            for (int count = 0; stream.Position < end && count < 100000; count++)
            {
                if (end - stream.Position < 8) throw Invalid("Truncated WAV chunk header.");
                stream.ReadExactly(chunk);
                uint size = BinaryPrimitives.ReadUInt32LittleEndian(chunk.AsSpan(4));
                long next = stream.Position + size + (size & 1);
                if (next > end) throw Invalid("WAV chunk exceeds RIFF extent.");
                if (chunk.AsSpan(0, 4).SequenceEqual("fmt "u8))
                {
                    if (tag != 0 || size < 16 || size > 4096) throw Invalid("Invalid or duplicate WAV format.");
                    var fmt = new byte[(int)size];
                    stream.ReadExactly(fmt);
                    tag = BinaryPrimitives.ReadUInt16LittleEndian(fmt);
                    if (tag is not (6 or 7 or 65534)) return null;
                    if (tag == 65534)
                    {
                        if (size < 40) throw Invalid("Truncated extensible WAV format.");
                        tag = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(24));
                        if (tag is not (6 or 7)) return null;
                        int extensionBytes = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(16));
                        int samplesOrPrecision = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(18));
                        if (extensionBytes < 22 || extensionBytes > size - 18 ||
                            samplesOrPrecision is not (0 or 1 or 8) ||
                            !fmt.AsSpan(26, 14).SequenceEqual(new byte[] { 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113 }))
                            throw Invalid("Invalid extensible G.711 subformat.");
                    }
                    channels = BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(2));
                    uint hz = BinaryPrimitives.ReadUInt32LittleEndian(fmt.AsSpan(4));
                    if (channels is < 1 or > 32 || hz is < 1 or > 192000 ||
                        BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(12)) != channels ||
                        BinaryPrimitives.ReadUInt16LittleEndian(fmt.AsSpan(14)) != 8 ||
                        BinaryPrimitives.ReadUInt32LittleEndian(fmt.AsSpan(8)) != hz * channels)
                        throw Invalid("Invalid G.711 sample format or block alignment.");
                    rate = (int)hz;
                }
                else if (chunk.AsSpan(0, 4).SequenceEqual("data"u8))
                {
                    if (tag == 0) throw Invalid("WAV data precedes its format.");
                    if (size % channels != 0) throw Invalid("G.711 data ends within a frame.");
                    return new G711WaveDecoder(stream, rate, channels, stream.Position, size, tag == 7);
                }
                stream.Position = next;
            }
            throw Invalid("WAV data is missing or too many chunks were supplied.");
        }
        catch (EndOfStreamException ex) { throw new UnsupportedAudioException("Truncated G.711 WAV.", ex); }
    }

    internal static short Expand(byte code, bool muLaw)
    {
        // ITU-T G.711: complement the transmitted code, then expand the segment and quantization fields.
        int value = code ^ (muLaw ? 255 : 85);
        int segment = (value >> 4) & 7;
        int magnitude;
        if (muLaw)
        {
            magnitude = (((value & 15) << 3) + 132) << segment;
            magnitude -= 132;
            return (short)((value & 128) != 0 ? -magnitude : magnitude);
        }
        magnitude = (value & 15) << 4;
        magnitude += segment == 0 ? 8 : 264;
        if (segment > 1) magnitude <<= segment - 1;
        return (short)((value & 128) != 0 ? magnitude : -magnitude);
    }

    public AudioFormat Format { get; }
    public long LengthFrames { get; }
    public long PositionFrames { get; private set; }
    public bool CanSeek => true;
    public bool HasNativeIntegers => false;
    public int Read(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        int frames = (int)Math.Min(Math.Min(destination.Length / Format.Channels, bytes.Length / Format.Channels), LengthFrames - PositionFrames);
        int samples = frames * Format.Channels;
        try { stream.ReadExactly(bytes.AsSpan(0, samples)); }
        catch (EndOfStreamException ex) { throw new UnsupportedAudioException("Truncated G.711 payload.", ex); }
        for (int i = 0; i < samples; i++) destination[i] = Expand(bytes[i], muLaw) / 32768f;
        PositionFrames += frames;
        return frames;
    }
    public int ReadInteger(Span<int> destination) => throw new NotSupportedException("G.711 is companded, not linear PCM.");
    public void Seek(long frame)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (frame < 0 || frame > LengthFrames) throw new ArgumentOutOfRangeException(nameof(frame));
        stream.Position = dataStart + frame * Format.Channels;
        PositionFrames = frame;
    }
    public void Dispose() => disposed = true;
    private static UnsupportedAudioException Invalid(string message) => new(message);
}
