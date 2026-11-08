using OneSpeech.Embedders;
using Xunit;

namespace OneSpeech.Tests;

public sealed class CampPlusEmbedderTests
{
    [Fact]
    public void CallerBufferLifetimeCancellationAndValidation()
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onemodel");
        var graph = new SpeakerGraph();
        float[] basis = new float[512]; basis[0] = 1;
        graph.Save(path, graph.Float("basis", [1, 512], basis), 512);
        try
        {
            // Algebraic constant-output test graph, not a voice or biometric fixture.
            var info = new ModelInfo { Id = "algebraic", Source = "https://example.org", Revision = "none", License = "MIT", Attribution = "Algebraic constant test.", Tier = ModelTier.Tier1 };
            using var embedder = new CampPlusEmbedder(path, info);
            float[] output = new float[512];
            var valid = new AudioBuffer(new float[16000], 16000);
            embedder.Embed(valid, output, CancellationToken.None);
            Assert.Equal(1, output[0]); Assert.Equal(0, output[1]);
            Assert.Throws<ArgumentException>(() => embedder.Embed(new AudioBuffer(new float[400], 16000), output, CancellationToken.None));
            Assert.Throws<ArgumentException>(() => embedder.Embed(new AudioBuffer(new float[16000], 8000), output, CancellationToken.None));
            Assert.Throws<ArgumentException>(() => embedder.Embed(valid, new float[192], CancellationToken.None));
            Assert.Throws<OperationCanceledException>(() => embedder.Embed(valid, output, new CancellationToken(true)));
            embedder.Dispose(); embedder.Dispose();
            Assert.Throws<ObjectDisposedException>(() => embedder.Embed(valid, output, CancellationToken.None));
        }
        finally { File.Delete(path); }
    }
}
