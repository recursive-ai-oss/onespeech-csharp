using OneSpeech.Catalog.Conversion;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;
using Xunit;

namespace OneSpeech.Tests;

public sealed class NemotronSubsamplingTests
{
    // The first five rows are physical mel/encoder lengths observed through NeMo 3.0.0's
    // public encoder API; the remaining rows exercise the same padding rule at short edges.
    [Theory]
    [InlineData(1044, 132)]
    [InlineData(328, 42)]
    [InlineData(663, 84)]
    [InlineData(269, 35)]
    [InlineData(522, 67)]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(7, 2)]
    [InlineData(8, 2)]
    public void EndPaddingAtEachStrideMatchesNativePhysicalLengths(int frames, int expected)
    {
        var dimensions = new FastConformerDimensions(MelBins: 16, Width: 8, Layers: 2, Heads: 2, Expansion: 2, ConvolutionKernel: 3, SubsamplingChannels: 32);
        var source = new FastConformerGraphTests.SyntheticTensors(dimensions, causal: true);
        string directory = Path.Combine(Path.GetTempPath(), "onespeech-native-length-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var ct = TestContext.Current.CancellationToken;
        try
        {
            string path = Path.Combine(directory, "subsample.onemodel");
            new NemotronAsrGraphBuilder(source, dimensions).Subsampling(path, false, ct);
            using var session = Session.Load(path, new ModelOptions { IntraOpThreads = 1 }, cancellationToken: ct);
            using var input = Tensor.Zeros(DType.F32, Shape.Create(1, 16, frames));
            using var result = session.Run(new Dictionary<string, Tensor> { ["features"] = input });
            Assert.Equal(8 * expected, result["encoded"].Data<float>().Length);
        }
        finally { Directory.Delete(directory, true); }
    }
}
