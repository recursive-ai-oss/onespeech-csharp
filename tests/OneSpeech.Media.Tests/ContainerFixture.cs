using OneAudio.Aac;
using OneAudio.Mp4;
using RecursiveAI.OneMedia;
using RecursiveAI.OneMedia.Matroska;

namespace OneSpeech.Media.Tests;

internal static class ContainerFixture
{
    // Deterministic analytic tone, no speaker data; the optional speech clip is the repository's attributed LibriSpeech fixture.
    internal static MemoryStream Aac(string? clip = null)
    {
        float[] samples;
        if (clip is not null)
        {
            using var source = OneAudio.Audio.Open(clip);
            var list = new List<float>();
            var buffer = new float[4096];
            int read;
            while ((read = source.Read(buffer)) != 0) list.AddRange(buffer.AsSpan(0, read).ToArray());
            samples = [.. list];
        }
        else samples = Enumerable.Range(0, 16000).Select(i => (float)(0.02 * Math.Sin(2 * Math.PI * 220 * i / 16000))).ToArray();
        var output = new MemoryStream();
        var encoder = new AacEncoder(16000, 1);
        using (var writer = Mp4AudioWriter.CreateAac(output, 16000, 1, encoder.AudioSpecificConfig, AacEncoder.PrimingFrames, leaveOpen: true))
        {
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
        output.Position = 0;
        return output;
    }

    internal static MemoryStream Matroska(Stream aac, CancellationToken cancellationToken)
    {
        using var media = RecursiveAI.OneMedia.Media.Open(aac, leaveOpen: true, cancellationToken);
        var output = new MemoryStream();
        using (var muxer = new MatroskaMuxer(output))
        {
            // An empty synthetic video track proves audio selection does not rely on track zero being audio. No video decoder is exercised.
            muxer.AddTrack(new MediaTrack { Index = 0, Kind = TrackKind.Video, Codec = CodecId.Vp8, Timescale = 1000, Video = new(16, 16) });
            int index = muxer.AddTrack(media.Audio!);
            foreach (var packet in media.Packets(media.Audio!))
                using (packet) muxer.WritePacket(packet.Info with { TrackIndex = index }, packet.Data.Span, cancellationToken);
            muxer.Finish(cancellationToken);
        }
        output.Position = 0;
        return output;
    }
}
