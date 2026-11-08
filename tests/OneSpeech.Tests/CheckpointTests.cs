using System.IO.Compression;
using System.Text;
using Xunit;

namespace OneSpeech.Tests;

public sealed class CheckpointTests
{
    [Fact]
    public void TensorDataIsReadWithoutExecutingGlobals()
    {
        using var stream = Fixture();
        var tensors = TensorCheckpoint.Read(stream);
        Assert.Equal(2, Assert.Single(tensors["weight"].Shape));
        Assert.Equal(new float[] { 1, 2 }, tensors["weight"].Values);
    }

    [Fact]
    public void HostileTensorMetadataAndGlobalsAreRefused()
    {
        using var outside = Fixture(offset: 2);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(outside));
        using var negative = Fixture(dimension: -1);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(negative));
        using var huge = Fixture(dimension: 10000001);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(huge));
        using var executable = Zip([0x80, 2, .. Encoding.ASCII.GetBytes("cos\nsystem\n.")]);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(executable));
        using var shortOperand = Zip(new byte[] { 0x80, 2, 0x8a, 8, 1 });
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(shortOperand));
        using var truncated = new MemoryStream(new byte[] { 1, 2 });
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(truncated));
    }

    [Fact]
    public void NoncontiguousAndEmptyViewsHaveExplicitBounds()
    {
        using var reversedStride = Fixture(dimension: 2, stride: -1);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(reversedStride));
        using var strideTooLarge = Fixture(dimension: 2, stride: 2);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(strideTooLarge));
        using var empty = Fixture(offset: 2, dimension: 0);
        Assert.Empty(TensorCheckpoint.Read(empty)["weight"].Values);
        using var strided = Fixture(dimension: 2, stride: 0);
        Assert.All(TensorCheckpoint.Read(strided)["weight"].Values, value => Assert.Equal(1, value));
    }

    [Fact]
    public void InvalidUtf8AndDuplicateZipEntriesAreRefused()
    {
        using var invalidText = Zip([0x80, 2, (byte)'X', 1, 0, 0, 0, 0xff, (byte)'.']);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(invalidText));
        using var stream = Fixture();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Update, leaveOpen: true))
            zip.CreateEntry("archive/data.pkl");
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(stream));
    }

    [Theory]
    [InlineData(4, 1)]
    [InlineData(8, 0xff)]
    [InlineData(10, 0xff)]
    [InlineData(15, 0x7f)]
    [InlineData(19, 0xff)]
    public void OversizedOrSplitCentralDirectoriesAreRefused(int fieldOffset, byte value)
    {
        using var fixture = Fixture();
        byte[] bytes = fixture.ToArray();
        bytes[bytes.Length - 22 + fieldOffset] = value;
        using var hostile = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => TensorCheckpoint.Read(hostile));
    }

    // Synthetic tensor-only checkpoint: generated here, no speech or upstream implementation fixture.
    internal static MemoryStream Fixture(int offset = 0, int dimension = 2, int stride = 1)
    {
        using var data = new MemoryStream();
        using (var writer = new BinaryWriter(data, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(new byte[] { 0x80, 2, (byte)'}', (byte)'(' });
            Text(writer, "weight");
            Global(writer, "torch._utils", "_rebuild_tensor_v2");
            writer.Write((byte)'(');
            writer.Write((byte)'('); Text(writer, "storage"); Global(writer, "torch", "FloatStorage");
            Text(writer, "0"); Text(writer, "cpu"); Integer(writer, 2);
            writer.Write((byte)'t'); writer.Write((byte)'Q'); Integer(writer, offset);
            Integer(writer, dimension); writer.Write((byte)0x85);
            Integer(writer, stride); writer.Write((byte)0x85);
            writer.Write((byte)0x89); writer.Write((byte)'}');
            writer.Write((byte)'t'); writer.Write((byte)'R'); writer.Write((byte)'u'); writer.Write((byte)'.');
        }
        return Zip(data.ToArray());
    }

    private static MemoryStream Zip(byte[] pickle)
    {
        var result = new MemoryStream();
        using (var zip = new ZipArchive(result, ZipArchiveMode.Create, leaveOpen: true))
        {
            using (var index = zip.CreateEntry("archive/data.pkl").Open()) index.Write(pickle);
            using var storage = zip.CreateEntry("archive/data/0").Open();
            using var writer = new BinaryWriter(storage);
            writer.Write(1f); writer.Write(2f);
        }
        result.Position = 0;
        return result;
    }
    private static void Text(BinaryWriter writer, string value)
    {
        writer.Write((byte)'X'); var bytes = Encoding.UTF8.GetBytes(value); writer.Write(bytes.Length); writer.Write(bytes);
    }
    private static void Integer(BinaryWriter writer, int value) { writer.Write((byte)'J'); writer.Write(value); }
    private static void Global(BinaryWriter writer, string module, string name)
    {
        writer.Write((byte)'c'); writer.Write(Encoding.ASCII.GetBytes(module + "\n" + name + "\n"));
    }
}
