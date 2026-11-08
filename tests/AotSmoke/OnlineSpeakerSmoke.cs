using OneSpeech;
using OneSpeech.Live;

internal static class OnlineSpeakerSmoke
{
    internal static void Run()
    {
        using var inner = new EndpointSession();
        using var model = new BasisEmbedder();
        using var live = new OnlineDiarizationSession(inner, model);
        var samples = new float[16000];
        Array.Fill(samples, 1);
        var first = live.Push(new AudioBuffer(samples, 16000)).OfType<FinalEvent>().Single();
        Array.Fill(samples, -1);
        var second = live.Push(new AudioBuffer(samples, 16000)).OfType<FinalEvent>().Single();
        if (first.Segment.Words[0].Speaker?.Id != "speaker-1" || second.Segment.Words[0].Speaker?.Id != "speaker-2")
            throw new InvalidOperationException("Online speaker NativeAOT smoke failed.");
        if (first.SpeakerTurns.Single().Speaker.Id != "speaker-1" || second.SpeakerTurns.Single().Speaker.Id != "speaker-2")
            throw new InvalidOperationException("Final speaker activity NativeAOT smoke failed.");
        live.Complete();
        Console.WriteLine("Two-speaker live NativeAOT smoke passed.");
    }
    private sealed class EndpointSession : LiveSession
    {
        private TimeSpan start;
        protected override void Process(AudioBuffer audio, CancellationToken cancellationToken)
        {
            PublishFinal(new Segment { Text = "speech", Start = start, End = AudioTime,
                Words = [new Word { Text = "speech", Start = start, End = AudioTime }] });
            PublishEndpoint(AudioTime, EndpointReason.DurationLimit); start = AudioTime;
        }
        protected override void Finish(CancellationToken cancellationToken) { }
    }
    private sealed class BasisEmbedder : SpeakerEmbedder
    {
        public override ModelInfo Model => throw new NotSupportedException();
        public override int Dimension => 2;
        public override int SampleRate => 16000;
        public override void Embed(AudioBuffer audio, Span<float> destination, CancellationToken cancellationToken)
        { destination[0] = audio.Samples.Span[0] > 0 ? 1 : 0; destination[1] = audio.Samples.Span[0] > 0 ? 0 : 1; }
        public override void Dispose() { }
    }
}
