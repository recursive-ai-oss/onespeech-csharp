using OneSpeech;
using OneSpeech.Audio;
using SharpFuzz;

Fuzzer.LibFuzzer.Run(data =>
{
    if (data.Length < 12 || !data[..4].SequenceEqual("RIFF"u8) || !data.Slice(8, 4).SequenceEqual("WAVE"u8)) return;
    try
    {
        using var stream = new MemoryStream(data.ToArray());
        using var source = AudioFileSource.Open(stream, true, CancellationToken.None);
        var pcm = new float[2048 * source.Channels];
        _ = source.Read(pcm, CancellationToken.None);
    }
    catch (UnsupportedAudioException) { }
});
