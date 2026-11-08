using RecursiveAI.OneMedia;
using RecursiveAI.OneMedia.Captions;
using RecursiveAI.OneMedia.IsoBmff;

namespace OneSpeech.Media;

/// <summary>Text-track encoding and MP4 caption muxing through OneMedia. Does not encode video or audio.</summary>
public static class CaptionMuxing
{
    /// <summary>Encodes captions to a container text track, WebVTT-in-MP4 by default. Returns OneMedia's packets, declaration and loss report. Declare every track before writing any packets; set packet TrackIndex to the index returned by the muxer. Unknown-time cues are dropped and reported by OneMedia. Cancellation is checked during caption conversion.</summary>
    public static TextTrackEncoding EncodeCaptionTrack(this Transcript transcript, TextTrackFormat format = TextTrackFormat.WebVttIso, CancellationToken cancellationToken = default) =>
        TextTrackEncoder.Encode(transcript.ToSubtitles(cancellationToken), format);

    /// <summary>Copies compatible tracks from an MP4/MOV/3GP input into a new MP4 and adds a WebVTT caption track. Opens with OneMedia; supported codecs/configurations are those IsoMuxer accepts, with no codec conversion or metadata copying. Input/output must differ and output must not exist. Packets are copied one track at a time in decode order, without re-encoding. Uses seekable moov-last output to avoid payload spooling. On failure deletes only the newly created incomplete output. Returns the caption encoding including fidelity losses. Throws NotSupportedException for non-ISO input, OneMedia's typed format/mux errors, IOException or OperationCanceledException. Checks cancellation per packet.</summary>
    public static TextTrackEncoding MuxCaptions(this Transcript transcript, string inputPath, string outputPath, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        ArgumentException.ThrowIfNullOrEmpty(inputPath);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        cancellationToken.ThrowIfCancellationRequested();
        var encoded = transcript.EncodeCaptionTrack(cancellationToken: cancellationToken);
        using var input = RecursiveAI.OneMedia.Media.Open(inputPath, cancellationToken);
        if (input.Format != MediaFormatId.IsoBmff) throw new NotSupportedException("Caption copy-muxing requires MP4/MOV/3GP input to preserve packet framing. Use OneMedia's remuxer for container conversion, or EncodeCaptionTrack with a caller-managed muxer.");
        using var output = new FileStream(outputPath, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None);
        try
        {
            using var muxer = new IsoMuxer(output, new IsoMuxOptions { Layout = IsoLayout.MoovLast });
            var tracks = input.Tracks.Select(track => (Track: track, Index: muxer.AddTrack(track))).ToArray();
            int subtitleIndex = muxer.AddTrack(encoded.Track);
            foreach (var track in tracks)
                foreach (var packet in input.Packets(track.Track))
                    using (packet)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        muxer.WritePacket(packet.Info with { TrackIndex = track.Index }, packet.Data.Span, cancellationToken);
                    }
            foreach (var packet in encoded.Packets)
            {
                cancellationToken.ThrowIfCancellationRequested();
                muxer.WritePacket(packet.Info with { TrackIndex = subtitleIndex }, packet.Data, cancellationToken);
            }
            muxer.Finish(cancellationToken);
            return encoded;
        }
        catch
        {
            output.Dispose();
            File.Delete(outputPath);
            throw;
        }
    }
}
