using System.Buffers.Binary;
using System.Text;
using OneAudio.Aac;
using OneAudio.Mp4;
using OneSpeech.Audio;
using OneSpeech.Vad;
using Xunit;

namespace OneSpeech.Tests;

public sealed class VoicemailInputTests
{
    [Theory]
    [InlineData("amr-nb", 32000)]
    [InlineData("amr-wb", 32000)]
    [InlineData("3gp", 32000)]
    [InlineData("m4a", 32000)]
    [InlineData("mu-law", 32000)]
    [InlineData("a-law", 32000)]
    public void SyntheticVoicemailDecodesThroughVad(string format, int expectedFrames)
    {
        using var encoded = format switch
        {
            "amr-nb" => Amr(false), "amr-wb" => Amr(true), "3gp" => ThreeGp(), "m4a" => Aac(),
            "mu-law" => AudioPipelineTests.Wave(7, Enumerable.Repeat((byte)255, 16000).ToArray(), rate: 8000),
            _ => AudioPipelineTests.Wave(6, Enumerable.Repeat((byte)213, 16000).ToArray(), rate: 8000)
        };
        var token = TestContext.Current.CancellationToken;
        using var source = AudioFileSource.Open(encoded, true, token);
        using var vad = new SileroVad(cancellationToken: token);
        int total = 0;
        IEnumerable<AudioBuffer> Blocks()
        {
            var pcm = new float[1024 * source.Channels];
            int frames;
            while ((frames = source.Read(pcm, token)) > 0)
            {
                Assert.All(pcm[..(frames * source.Channels)], sample => Assert.True(float.IsFinite(sample)));
                total += frames;
                yield return new AudioBuffer(pcm.AsMemory(0, frames * source.Channels), 16000, source.Channels);
            }
        }
        var chunks = new VadChunker(vad).Read(Blocks(), null, token).ToArray();
        Assert.Equal(expectedFrames, total);
        Assert.All(chunks, c => { Assert.Equal(16000, c.Audio.SampleRate); Assert.Equal(1, c.Audio.Channels); Assert.True(c.Audio.Duration <= TimeSpan.FromSeconds(30)); });
    }

    // Synthetic storage frames: fixed arithmetic payload, not a recording or a trained voice generator.
    private static MemoryStream Amr(bool wide)
    {
        var file = new MemoryStream();
        file.Write(wide ? "#!AMR-WB\n"u8 : "#!AMR\n"u8);
        for (int i = 0; i < 100; i++)
        {
            file.WriteByte(4);
            for (int k = 0; k < (wide ? 17 : 12); k++) file.WriteByte((byte)(i * 5 + k));
        }
        file.Position = 0;
        return file;
    }

    private static MemoryStream Aac()
    {
        var stream = new MemoryStream();
        var encoder = new AacEncoder(16000, 1);
        using (var writer = Mp4AudioWriter.CreateAac(stream, 16000, 1, encoder.AudioSpecificConfig, AacEncoder.PrimingFrames, leaveOpen: true))
        {
            var samples = new float[32000];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = (float)(0.08 * Math.Sin(i * 2 * Math.PI * 170 / 16000) * (1 + Math.Sin(i * 2 * Math.PI * 4 / 16000)));
            int offset = 0;
            while (offset < samples.Length)
            {
                offset += encoder.Write(samples.AsSpan(offset));
                while (encoder.TryReadPacket(out var packet)) writer.AddSample(packet.Data.Span, 1024);
            }
            encoder.Finish();
            while (encoder.TryReadPacket(out var packet)) writer.AddSample(packet.Data.Span, 1024);
            writer.Finish(samples.Length);
        }
        stream.Position = 0;
        return stream;
    }

    // Derived from the family's OneAudio ContainerSmoke synthetic ISO BMFF fixture builder.
    private static MemoryStream ThreeGp()
    {
        using var raw = Amr(false);
        byte[] media = raw.ToArray()[6..];
        var ftyp = Box("ftyp", "3gp4"u8.ToArray(), U32(0), "3gp4"u8.ToArray());
        var mdat = Box("mdat", media);
        var samr = Box("samr", new byte[6], new byte[] { 0, 1 }, new byte[8], new byte[] { 0, 1, 0, 16 }, new byte[4], U32(8000u << 16),
            Box("damr", "test"u8.ToArray(), new byte[] { 0, 0x81, 0xFF, 0, 1 }));
        var stbl = Box("stbl", Box("stsd", U32(0, 1), samr), Box("stts", U32(0, 1, 100, 160)),
            Box("stsc", U32(0, 1, 1, 100, 1)), Box("stsz", U32(0, 13, 100)), Box("stco", U32(0, 1, (uint)(ftyp.Length + 8))));
        var mdia = Box("mdia", Box("mdhd", U32(0, 0, 0, 8000, 16000, 0)), Box("hdlr", U32(0, 0), "soun"u8.ToArray(), new byte[13]), Box("minf", Box("smhd", new byte[8]), stbl));
        var moov = Box("moov", Box("mvhd", U32(0, 0, 0, 1000, 2000, 0x10000)), Box("trak", Box("tkhd", U32(7, 0, 0, 1)), mdia));
        return new MemoryStream([.. ftyp, .. mdat, .. moov]);
    }

    private static byte[] U32(params uint[] values)
    {
        var result = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteUInt32BigEndian(result.AsSpan(i * 4), values[i]);
        return result;
    }
    private static byte[] Box(string type, params byte[][] parts)
    {
        int length = 8 + parts.Sum(p => p.Length);
        return [.. U32((uint)length), .. Encoding.ASCII.GetBytes(type), .. parts.SelectMany(p => p)];
    }
}
