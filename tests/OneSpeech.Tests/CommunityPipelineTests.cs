using System.IO.Compression;
using System.Text;
using OneSpeech.Catalog;
using OneSpeech.Diarization;
using OneSpeech.Formats;
using Xunit;

namespace OneSpeech.Tests;

public sealed class CommunityPipelineTests
{
    private static byte[] Npy(string descr, string shape, byte[] payload, string order = "False")
    {
        string header = $"{{'descr': '{descr}', 'fortran_order': {order}, 'shape': {shape}, }}";
        header = header.PadRight((64 - (10 + header.Length + 1) % 64) % 64 + header.Length) + "\n";
        var bytes = new List<byte> { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0, (byte)(header.Length & 255), (byte)(header.Length >> 8) };
        bytes.AddRange(Encoding.ASCII.GetBytes(header)); bytes.AddRange(payload);
        return [.. bytes];
    }

    private static byte[] Doubles(params double[] values) => values.SelectMany(BitConverter.GetBytes).ToArray();

    [Fact]
    public void NpyArraysAreReadWithShapeAndValues()
    {
        var a = NpzArchive.ParseNpy(Npy("<f8", "(2, 2)", Doubles(1, 2, 3, 4)));
        Assert.Equal([2, 2], a.Shape);
        Assert.Equal([1d, 2, 3, 4], a.Values);
        Assert.Equal([3], NpzArchive.ParseNpy(Npy("<f8", "(3,)", Doubles(1, 2, 3))).Shape);
    }

    [Fact]
    public void NpzMembersAreReadFromZip()
    {
        using var zip = new MemoryStream();
        using (var archive = new ZipArchive(zip, ZipArchiveMode.Create, leaveOpen: true))
        {
            using var s = archive.CreateEntry("mu.npy").Open();
            s.Write(Npy("<f8", "(2,)", Doubles(5, 6)));
        }
        zip.Position = 0;
        Assert.Equal([5d, 6], NpzArchive.Read(zip)["mu"].Values);
    }

    [Theory]
    [InlineData("<f8", "(3,)", 16, "False")]       // payload shorter than the shape says
    [InlineData("<f8", "(1000000000, 1000000000)", 8, "False")] // huge count
    [InlineData("<f8", "(2, 2)", 32, "True")]       // Fortran order
    [InlineData("<c16", "(1,)", 16, "False")]       // unsupported dtype
    [InlineData("<f8", "(1, 1, 1, 1, 1)", 8, "False")] // too many dimensions
    public void HostileNpyHeadersAreRejected(string descr, string shape, int payloadBytes, string order)
        => Assert.Throws<InvalidDataException>(() => NpzArchive.ParseNpy(Npy(descr, shape, new byte[payloadBytes], order)));

    [Fact]
    public void TruncatedAndGarbageNpyAreRejected()
    {
        byte[] good = Npy("<f8", "(2,)", Doubles(1, 2));
        for (int cut = 0; cut < good.Length - 1; cut += 7) Assert.Throws<InvalidDataException>(() => NpzArchive.ParseNpy(good.AsSpan(0, cut)));
        Assert.Throws<InvalidDataException>(() => NpzArchive.ParseNpy(new byte[64]));
    }

    [Fact]
    public void FrameGridMatchesTheObservedReference()
    {
        // 180 s and 100.3 s excerpts: 171 and 92 windows gave 10668 and 5986 frames in the reference output.
        Assert.Equal(171, CommunityPipeline.ChunkCount(180 * 16000));
        Assert.Equal(92, CommunityPipeline.ChunkCount((int)(100.3 * 16000)));
        Assert.Equal(1, CommunityPipeline.ChunkCount(5 * 16000));
        Assert.Equal(10668, CommunityPipeline.TotalFrames(171));
        Assert.Equal(5986, CommunityPipeline.TotalFrames(92));
        Assert.Equal(59, CommunityPipeline.Offset(1));
    }

    [Fact]
    public void CentroidLinkageStopsAtTheThreshold()
    {
        float[][] points = [[0, 0], [0.1f, 0], [5, 5], [5, 5.1f], [20, 0]];
        Assert.Equal([0, 0, 1, 1, 2], CentroidLinkage.Cluster(points, 1.0));
        Assert.Equal([0, 0, 0, 0, 1], CentroidLinkage.Cluster(points, 8.0));
        Assert.Empty(CentroidLinkage.Cluster([], 1.0));
    }

    [Fact]
    public void VbxMergesARedundantSpeakerAndKeepsRealOnes()
    {
        // Two well-separated speakers in blocks, over-split into four initial clusters.
        var rng = new Random(7);
        int per = 60, dim = 8;
        var phi = Enumerable.Repeat(4.0, dim).ToArray();
        var x = new List<double[]>(); var init = new List<int>();
        for (int block = 0; block < 4; block++)
            for (int i = 0; i < per; i++)
            {
                int speaker = block % 2;
                var v = new double[dim];
                for (int d = 0; d < dim; d++) v[d] = (speaker == 0 ? 3.0 : -3.0) + (rng.NextDouble() - 0.5);
                x.Add(v); init.Add(block);
            }
        var result = VbxClustering.Run([.. x], phi, [.. init], 4, 0.07, 0.8, 0.5, 7, 400, 1e-9);
        int alive = result.Priors.Count(p => p > 1e-7);
        Assert.Equal(2, alive);
        var hard = x.Select((_, t) => Array.IndexOf(result.Responsibilities[t], result.Responsibilities[t].Max())).ToArray();
        Assert.Equal(hard[0], hard[per * 2]);        // blocks 0 and 2 are the same speaker
        Assert.NotEqual(hard[0], hard[per]);
    }

    [Fact]
    public void ReconstructionKeepsTheTopClustersByCount()
    {
        // Two windows, one local speaker active, both with the same embedding: one cluster, count 1 where active.
        int chunks = 2;
        var masks = new byte[chunks][];
        for (int c = 0; c < chunks; c++) { masks[c] = new byte[589 * 3]; for (int f = 100; f < 300; f++) masks[c][f * 3] = 1; }
        var embeddings = new float[chunks * 3][];
        embeddings[0] = Enumerable.Range(0, 256).Select(i => i % 3 == 0 ? 1f : 0.1f).ToArray();
        embeddings[3] = (float[])embeddings[0]!.Clone();
        var plda = PldaTransform.Create(IdentityPlda(), IdentityTransform());
        var pipeline = new CommunityPipeline(null!, null!, plda, new CommunityOptions());
        var result = pipeline.Finish(chunks, 11 * 16000, masks, embeddings);
        Assert.Single(result.Centroids);
        Assert.Equal(CommunityPipeline.TotalFrames(2), result.Counts.Length);
        Assert.Equal(1, result.Counts[200]);
        Assert.Equal(0, result.Counts[50]);
        Assert.Single(result.Turns);
        Assert.True(result.Turns[0].Start < result.Turns[0].End);
    }

    [Fact]
    public void EmbeddingsWithLittleOverlapFreeSpeechJoinTheNearestClusterWithoutShapingIt()
    {
        // Two windows with 200 overlap-free frames and one with 40 (under 0.2 of 589): the short one has a different
        // embedding, must still be assigned, and must not move the centroid.
        int chunks = 3;
        var masks = new byte[chunks][];
        for (int c = 0; c < chunks; c++) { masks[c] = new byte[589 * 3]; for (int f = 100; f < (c == 2 ? 140 : 300); f++) masks[c][f * 3] = 1; }
        var embeddings = new float[chunks * 3][];
        embeddings[0] = Enumerable.Range(0, 256).Select(i => i % 3 == 0 ? 1f : 0.1f).ToArray();
        embeddings[3] = (float[])embeddings[0]!.Clone();
        embeddings[6] = Enumerable.Range(0, 256).Select(i => i % 3 == 1 ? 1f : 0.1f).ToArray();
        var plda = PldaTransform.Create(IdentityPlda(), IdentityTransform());
        var result = new CommunityPipeline(null!, null!, plda, new CommunityOptions()).Finish(chunks, 12 * 16000, masks, embeddings);
        Assert.Single(result.Centroids);
        Assert.Equal(0, result.Assignment[6]);
        Assert.Equal(embeddings[0]![0], result.Centroids[0][0]);
        Assert.Equal(embeddings[0]![1], result.Centroids[0][1]);
    }

    [Fact]
    public void LocalSpeakersOfOneWindowNeverShareACluster()
    {
        var a = Enumerable.Range(0, 256).Select(i => i < 128 ? 1f : 0f).ToArray();
        var b = Enumerable.Range(0, 256).Select(i => i < 128 ? 0f : 1f).ToArray();
        // Both local speakers are closest to the first centre; the injective assignment must still separate them.
        var assignment = CommunityPipeline.Injective([CommunityPipeline.Normalize(a), CommunityPipeline.Normalize(a)], [CommunityPipeline.Normalize(a), CommunityPipeline.Normalize(b)]);
        Assert.Equal(2, assignment.Distinct().Count());
    }

    [Fact]
    public void LoadLocalChecksFilesBeforeAnyConversion()
    {
        string dir = Path.Combine(Path.GetTempPath(), "onespeech-community-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(dir, "segmentation"));
            File.WriteAllBytes(Path.Combine(dir, "segmentation", "pytorch_model.bin"), new byte[10]);
            Assert.Throws<ModelIntegrityException>(() => CommunityDiarizer.LoadLocal(dir, Path.Combine(dir, "cache"), 1, TestContext.Current.CancellationToken));
            Assert.ThrowsAny<IOException>(() => CommunityDiarizer.LoadLocal(Path.Combine(dir, "missing"), Path.Combine(dir, "cache"), 1, TestContext.Current.CancellationToken));
            Assert.Throws<ArgumentOutOfRangeException>(() => CommunityDiarizer.LoadLocal(dir, null, 0, TestContext.Current.CancellationToken));
            Assert.False(Directory.Exists(Path.Combine(dir, "cache")));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void CatalogPinsAllFourCommunityFilesAsTierTwoGated()
    {
        var entry = ModelCatalog.Default.Get("pyannote/speaker-diarization-community-1");
        Assert.Equal(["segmentation/pytorch_model.bin", "embedding/pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"], entry.Files.Select(f => f.Name));
        Assert.True(entry.Gated);
        Assert.Equal(ModelTier.Tier2, entry.Tier);
        Assert.False(entry.IsDefault);
        Assert.All(entry.Files, f => Assert.StartsWith("https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/", f.UpstreamUrl, StringComparison.Ordinal));
    }

    private static Dictionary<string, NpyArray> IdentityPlda()
    {
        var tr = new double[128 * 128]; for (int i = 0; i < 128; i++) tr[i * 128 + i] = 1;
        return new() { ["mu"] = new([128], new double[128]), ["tr"] = new([128, 128], tr), ["psi"] = new([128], Enumerable.Repeat(1.0, 128).ToArray()) };
    }

    private static Dictionary<string, NpyArray> IdentityTransform()
    {
        var lda = new double[256 * 128]; for (int i = 0; i < 128; i++) lda[i * 128 + i] = 1;
        return new() { ["mean1"] = new([256], new double[256]), ["mean2"] = new([128], new double[128]), ["lda"] = new([256, 128], lda) };
    }
}
