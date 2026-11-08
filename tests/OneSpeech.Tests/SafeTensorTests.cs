using System.Buffers.Binary;
using System.Text;
using OneSpeech.Formats;
using Xunit;

namespace OneSpeech.Tests;

public sealed class SafeTensorTests
{
    [Theory]
    [InlineData("F32", "0000803F000000C0")]
    [InlineData("F16", "003C00C0")]
    [InlineData("BF16", "803F00C0")]
    public void ReadsFloatTypesAndCopiesOriginalBytes(string dtype, string hex)
    {
        byte[] data = System.Convert.FromHexString(hex);
        using var stream = FileBytes($$$"""{"weight":{"dtype":"{{{dtype}}}","shape":[2],"data_offsets":[0,{{{data.Length}}}]},"__metadata__":{"format":"pt"}}""", data);
        using var file = new SafeTensorFile(stream, leaveOpen: true);
        Assert.Equal(new[] { 1f, -2f }, file.ReadFloat32("weight", CancellationToken.None));
        using var copy = new MemoryStream();
        file.CopyTensorTo("weight", copy, CancellationToken.None);
        Assert.Equal(data, copy.ToArray());
        using var expanded = new MemoryStream();
        file.CopyFloat32To("weight", expanded, CancellationToken.None);
        Assert.Equal(System.Convert.FromHexString("0000803F000000C0"), expanded.ToArray());
        Assert.Equal(2, file.Tensors["weight"].ElementCount);
        file.Dispose();
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void AllowsScalarsAndEmptyShapesAndOutOfOrderTensors()
    {
        using var stream = FileBytes("""{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"F32","shape":[0,9223372036854775807],"data_offsets":[0,0]}}""", new byte[4]);
        using var file = new SafeTensorFile(stream);
        Assert.Single(file.ReadFloat32("scalar", CancellationToken.None));
        Assert.Empty(file.ReadFloat32("empty", CancellationToken.None));
    }

    [Theory]
    [InlineData("{}", 1)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 3)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1,5]}}", 5)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,0]}}", 0)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,2],\"data_offsets\":[0,0]}}", 0)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4],\"data_offsets\":[0,4]}}", 4)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4)]
    [InlineData("{\"__metadata__\":{\"x\":1}}", 0)]
    [InlineData("{\"__metadata__\":{\"x\":\"a\",\"x\":\"b\"}}", 0)]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4,5]}}", 4)]
    [InlineData("{} {}", 0)]
    [InlineData(" { }", 0)]
    [InlineData("{", 0)]
    public void RejectsHostileHeaders(string header, int bytes)
    {
        using var stream = FileBytes(header, new byte[bytes]);
        Assert.Throws<InvalidDataException>(() => new SafeTensorFile(stream));
    }

    [Theory]
    [InlineData("{\"\\ud800\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}") ]
    [InlineData("{\"a\":{\"dtype\":\"\\ud800\",\"shape\":[],\"data_offsets\":[0,4]}}") ]
    public void RejectsUnpairedSurrogate(string header)
    {
        using var stream = FileBytes(header, new byte[4]);
        Assert.Throws<InvalidDataException>(() => new SafeTensorFile(stream));
    }

    [Fact]
    public void RejectsHugeLengthAndInvalidUtf8BeforeMaterializing()
    {
        using var stream = new MemoryStream(16);
        stream.SetLength(16);
        BinaryPrimitives.WriteUInt64LittleEndian(stream.GetBuffer(), ulong.MaxValue);
        Assert.Throws<InvalidDataException>(() => new SafeTensorFile(stream));
        using var invalid = FileBytes("{}", []);
        invalid.GetBuffer()[9] = 255;
        Assert.Throws<InvalidDataException>(() => new SafeTensorFile(invalid));
    }

    [Fact]
    public void ChecksCancellationAndOwnsTheStreamByDefault()
    {
        using var stream = FileBytes("""{"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}""", new byte[4]);
        using (var file = new SafeTensorFile(stream))
            Assert.Throws<OperationCanceledException>(() => file.ReadFloat32("a", new CancellationToken(true)));
        Assert.False(stream.CanRead);
    }

    private static MemoryStream FileBytes(string header, byte[] data)
    {
        byte[] json = Encoding.UTF8.GetBytes(header);
        var stream = new MemoryStream();
        byte[] prefix = new byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)json.Length);
        stream.Write(prefix);
        stream.Write(json);
        stream.Write(data);
        stream.Position = 0;
        return stream;
    }
}
