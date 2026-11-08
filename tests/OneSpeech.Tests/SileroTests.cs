using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using Xunit;

namespace OneSpeech.Tests;

public sealed class SileroTests
{
    [Fact]
    public void ReleasedGraphRuns()
    {
        using var stream = typeof(Speech).Assembly.GetManifestResourceStream("OneSpeech.silero_vad.onnx")!;
        using var session = Session.Load(stream, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: TestContext.Current.CancellationToken);
        using var input = Tensor.FromSpan<float>(new float[576], Shape.Create(1, 576));
        using var state = Tensor.FromSpan<float>(new float[256], Shape.Create(2, 1, 128));
        using var rate = Tensor.Scalar(16000L);
        using var result = session.Run(new Dictionary<string, Tensor> { ["input"] = input, ["state"] = state, ["sr"] = rate });
        Assert.InRange(result["output"].Data<float>()[0], 0, 1);
    }
}
