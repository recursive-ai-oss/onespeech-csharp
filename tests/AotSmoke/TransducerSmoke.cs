using OneSpeech.Audio;
using OneSpeech.Decoding;

internal static class TransducerSmoke
{
    internal static void Run()
    {
        var ctc = CtcGreedy.Decode([8, -8, -8, 8, 8, -8], 2, 1);
        if (ctc.Length != 2) throw new InvalidOperationException("CTC greedy smoke failed.");
        var beam = CtcPrefixBeamSearch.Decode([-.9f, -.5f, -.9f, -.5f], 2, 1);
        if (beam.Tokens.Length != 1) throw new InvalidOperationException("CTC beam smoke failed.");
        static void Joint(int frame, ReadOnlySpan<int> prefix, Span<float> tokens, Span<float> duration, CancellationToken ct)
        { tokens[0] = prefix.IsEmpty ? 10 : -10; tokens[1] = -tokens[0]; duration.Clear(); if (!duration.IsEmpty) duration[1] = 1; }
        var rnnt = RnntGreedy.Decode(2, 2, 1, Joint);
        var tdt = TdtGreedy.Decode(2, 2, 1, [0, 1], Joint);
        if (rnnt.Length != 1 || tdt.Length != 1) throw new InvalidOperationException("Transducer greedy smoke failed.");
        var words = NativeWordTiming.Map(tdt.AsSpan(), [new("hello", 0, 1)], TimeSpan.FromMilliseconds(80));
        if (words[0].End != TimeSpan.FromMilliseconds(80)) throw new InvalidOperationException("Native word timing smoke failed.");
        if (new NemoLogMelSpectrogram().Extract(new OneSpeech.AudioBuffer(new float[16000], 16000)).Length != 128 * 101)
            throw new InvalidOperationException("NeMo front end smoke failed.");
        Console.WriteLine("NeMo features, CTC, RNN-T, TDT and native timing NativeAOT smoke passed.");
    }
}
