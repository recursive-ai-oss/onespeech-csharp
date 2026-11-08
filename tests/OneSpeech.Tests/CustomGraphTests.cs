using OneSpeech.Catalog.Conversion;
using OneSpeech.Custom;
using OneSpeech.Formats;
using Xunit;

namespace OneSpeech.Tests;

public sealed class CustomGraphTests
{
    private static readonly ModelInfo Provenance = new() { Id = "synthetic", Revision = "1", Source = "synthetic", License = "MIT", Attribution = "Synthetic numerical fixture", Tier = ModelTier.Tier1 };

    [Theory]
    [InlineData(0)]
    [InlineData(4)]
    public void CtcGraphHasKnownTextAndTimings(int beam)
    {
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            string path = Path.Combine(root, "ctc.onemodel");
            Constant(path, 3, [1, 4, 3], [9, -9, -9, -9, -9, 9, -9, 9, -9, -9, -9, 9]);
            using var recognizer = new GraphSpeechRecognizer([new() { Path = path, Channels = 3, Stride = 100 }], GraphFrontEnd.RawWaveform, new(["a", " b", ""]), 2, TimeSpan.FromTicks(62500), Provenance, beamSize: beam, cancellationToken: TestContext.Current.CancellationToken);
            var t = recognizer.Transcribe(new(new float[400], 16000), "en", TestContext.Current.CancellationToken);
            Assert.Equal("a b", t.Text); Assert.Equal(2, t.Words.Length);
            Assert.Equal(TimeSpan.Zero, t.Words[0].Start); Assert.Equal(TimeSpan.FromTicks(62500), t.Words[0].End);
            Assert.Equal(TimeSpan.FromTicks(125000), t.Words[1].Start); Assert.Equal(TimeSpan.FromTicks(187500), t.Words[1].End);
            Assert.All(t.Words, w => { Assert.Equal(TimingMethod.Native, w.TimingMethod); Assert.True(w.Confidence > .99); });
            Assert.Throws<ArgumentException>(() => recognizer.Transcribe(new(new float[] { float.NaN }, 16000), null, TestContext.Current.CancellationToken));
            Assert.Throws<InvalidModelShapeException>(() => recognizer.Transcribe(new(new float[800], 16000), null, TestContext.Current.CancellationToken));
            recognizer.Dispose();
            Assert.Throws<ObjectDisposedException>(() => recognizer.Transcribe(new(new float[400], 16000), null, TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(root, true); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PackedTransducerGraphsRunBothSearchKinds(bool tdt)
    {
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            string encoder = Path.Combine(root, "encoder.onemodel"), predictor = Path.Combine(root, "predictor.onemodel"), joint = Path.Combine(root, "joint.onemodel");
            Constant(encoder, 3, [1, 2, 4], new float[8]);
            Constant(predictor, 2, [1, 4], new float[4], 4);
            Constant(joint, 2, [1, tdt ? 5 : 3], tdt ? [-9, -9, 9, -9, 9] : [-9, -9, 9], 4);
            using var recognizer = new GraphSpeechRecognizer([new() { Path = encoder, Channels = 2, Stride = 100 }], GraphFrontEnd.RawWaveform, new(["a", "b", ""]), 2, TimeSpan.FromTicks(62500), Provenance,
                new() { PredictorPath = predictor, JointPath = joint, StateSize = 4, EmbeddingSize = 2, Embeddings = new float[6], Durations = tdt ? [0, 1] : [] }, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Empty(recognizer.Transcribe(new(new float[400], 16000), null, TestContext.Current.CancellationToken).Words);
        }
        finally { Directory.Delete(root, true); }
    }

    [Fact]
    public void ShapeAndVocabularyErrorsAreTypedBeforeInference()
    {
        Assert.Throws<InvalidModelShapeException>(() => new TokenVocabulary([]));
        Assert.Throws<InvalidModelShapeException>(() => new GraphSpeechRecognizer([], GraphFrontEnd.RawWaveform, new([""]), 0, TimeSpan.FromMilliseconds(20), Provenance, cancellationToken: TestContext.Current.CancellationToken));
        string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            string graph = Path.Combine(root, "wrong.onemodel"); Constant(graph, 2, [1, 2], [0, 1], 2);
            Assert.Throws<InvalidModelShapeException>(() => new GraphSpeechRecognizer([new() { Path = graph, Channels = 2 }], GraphFrontEnd.RawWaveform, new(["", "a"]), 0, TimeSpan.FromMilliseconds(20), Provenance, cancellationToken: TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(root, true); }
    }

    private static void Constant(string path, int inputRank, long[] outputShape, float[] values, int inputChannels = 1)
    {
        var graph = new FastConformerGraph(new NoWeights());
        if (inputRank == 3) graph.Input("input", 1, inputChannels, "samples"); else graph.Input("input", 1, inputChannels);
        graph.Output(graph.Float(values, outputShape), "output", outputShape.Select(d => (object)(int)d).ToArray()); graph.Save(path, TestContext.Current.CancellationToken);
    }
    private sealed class NoWeights : IFloatTensorSource
    {
        public long[] GetShape(string name) => throw new InvalidOperationException();
        public void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken) => throw new InvalidOperationException();
    }
}
