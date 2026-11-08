using OneSpeech.Vad;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using Xunit;

namespace OneSpeech.Tests;

public sealed class SileroBufferTests
{
    [Fact]
    public void ReusedOutputTensorsMatchContainerApiAtBothRates()
    {
        var token = TestContext.Current.CancellationToken;
        var clip = OneAudio.Audio.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "1089-134686-0001.flac"), new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 }, token);
        using var detector = new SileroVad(cancellationToken: token);
        foreach (int rate in new[] { 8000, 16000 })
        {
            using var resource = typeof(SileroVad).Assembly.GetManifestResourceStream("OneSpeech.silero_vad.onnx")!;
            using var reference = Session.Load(resource, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: token);
            using var actual = detector.CreateState(rate);
            int context = rate / 250, step = 16000 / rate;
            using var input = Tensor.Zeros(DType.F32, Shape.Create(1, actual.FrameSamples + context));
            using var state = Tensor.Zeros(DType.F32, Shape.Create(2, 1, 128));
            using var sampleRate = Tensor.Scalar((long)rate);
            var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["input"] = input, ["state"] = state, ["sr"] = sampleRate };
            var frame = new float[actual.FrameSamples];
            for (int offset = 0; offset < clip.Samples.Length / step; offset += frame.Length)
            {
                token.ThrowIfCancellationRequested();
                for (int i = 0; i < frame.Length; i++) frame[i] = (offset + i) * step < clip.Samples.Length ? clip.Samples.Span[(offset + i) * step] : 0;
                frame.AsSpan().CopyTo(input.MutableData<float>()[context..]);
                using var expected = reference.Run(inputs);
                Assert.Equal(expected["output"].Data<float>()[0], actual.ProcessFrame(frame, token));
                expected["stateN"].Data<float>().CopyTo(state.MutableData<float>());
                frame.AsSpan(frame.Length - context).CopyTo(input.MutableData<float>());
            }
        }
    }
}
