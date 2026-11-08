using OneAudio;
using OneSpeech.Audio;
using RecursiveAI.OneMedia.Audio;
using MediaApi = RecursiveAI.OneMedia.Media;

namespace OneSpeech.Media;

/// <summary>Opt-in container decoding for all core path/stream pipelines. No registration occurs implicitly.</summary>
public static class MediaInput
{
    /// <summary>Registers OneMedia as the fallback for formats OneAudio does not open. Call once at startup before concurrent inference. Speech path/stream transcription, alignment and file streaming then accept OneMedia containers with a decodable first audio track; video is never decoded. Existing standalone audio and G.711 handling stays unchanged. Replaces AudioFileSource.AdditionalDecoder; does not open files or download models.</summary>
    public static void Register() => AudioFileSource.AdditionalDecoder = Open;

    /// <summary>Yields final segments from a caller-owned encoded media stream after asynchronously spooling to a seekable temporary file (512 MiB limit, bounded copy buffer). Call Register first. Core's streaming pipeline then decodes in bounded chunks; no whole-file diarization pass is performed. Optional language/model have the same meaning as Speech.TranscribeStream. Enumeration disposal or cancellation removes the spool and closes models, not the input stream. Throws UnsupportedAudioException on the size limit; core model/audio errors and I/O errors propagate. No pending final is emitted on early disposal.</summary>
    public static async IAsyncEnumerable<Segment> TranscribeStream(Stream stream, string? language = null, string? model = null, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var segment in Speech.TranscribeStream(stream, language, model, cancellationToken).ConfigureAwait(false)) yield return segment;
    }

    private static OwnedDecoder Open(Stream stream, CancellationToken cancellationToken)
    {
        var media = MediaApi.Open(stream, leaveOpen: true, cancellationToken: cancellationToken);
        try { return new OwnedDecoder(media.Decode(), media); }
        catch { media.Dispose(); throw; }
    }

    private sealed class OwnedDecoder(IAudioDecoder decoder, IDisposable media) : IAudioDecoder
    {
        public AudioFormat Format => decoder.Format;
        public long LengthFrames => decoder.LengthFrames;
        public long PositionFrames => decoder.PositionFrames;
        public bool CanSeek => decoder.CanSeek;
        public bool HasNativeIntegers => decoder.HasNativeIntegers;
        public int Read(Span<float> destination) => decoder.Read(destination);
        public int ReadInteger(Span<int> destination) => decoder.ReadInteger(destination);
        public void Seek(long frame) => decoder.Seek(frame);
        public void Dispose()
        {
            try { decoder.Dispose(); }
            finally { media.Dispose(); }
        }
    }
}
