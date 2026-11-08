using System.IO.Compression;
using System.Text;
using SharpFuzz;

if (args is ["seed", var directory])
{
    Directory.CreateDirectory(directory);
    using var pickle = new MemoryStream();
    using (var writer = new BinaryWriter(pickle, Encoding.UTF8, leaveOpen: true))
    {
        writer.Write(new byte[] { 0x80, 2, (byte)'}', (byte)'(' });
        Text(writer, "weight");
        Global(writer, "torch._utils", "_rebuild_tensor_v2");
        writer.Write((byte)'('); writer.Write((byte)'(');
        Text(writer, "storage"); Global(writer, "torch", "FloatStorage");
        Text(writer, "0"); Text(writer, "cpu"); writer.Write((byte)'K'); writer.Write((byte)2);
        writer.Write((byte)'t'); writer.Write((byte)'Q'); writer.Write((byte)'K'); writer.Write((byte)0);
        writer.Write((byte)'K'); writer.Write((byte)2); writer.Write((byte)0x85);
        writer.Write((byte)'K'); writer.Write((byte)1); writer.Write((byte)0x85);
        writer.Write((byte)0x89); writer.Write((byte)'}');
        writer.Write((byte)'t'); writer.Write((byte)'R'); writer.Write((byte)'u'); writer.Write((byte)'.');
    }
    byte[] bytes = pickle.ToArray();
    File.WriteAllBytes(Path.Combine(directory, "tensor.pkl"), bytes);
    using var archive = Wrap(bytes);
    File.WriteAllBytes(Path.Combine(directory, "tensor.zip"), archive.ToArray());
    return;
}

bool pickleMode = Environment.GetEnvironmentVariable("ONESPEECH_CHECKPOINT_PICKLE") == "1";
Fuzzer.LibFuzzer.Run(data =>
{
    try
    {
        using var stream = pickleMode ? Wrap(data) : new MemoryStream(data.ToArray(), writable: false);
        _ = TensorCheckpoint.Read(stream);
    }
    catch (InvalidDataException) { }
    catch (IOException) { }
});

static MemoryStream Wrap(ReadOnlySpan<byte> pickle)
{
    var result = new MemoryStream();
    using (var zip = new ZipArchive(result, ZipArchiveMode.Create, leaveOpen: true))
    {
        using (var index = zip.CreateEntry("archive/data.pkl", CompressionLevel.NoCompression).Open()) index.Write(pickle);
        using var storage = zip.CreateEntry("archive/data/0", CompressionLevel.NoCompression).Open();
        using var writer = new BinaryWriter(storage);
        writer.Write(1f); writer.Write(2f);
    }
    result.Position = 0;
    return result;
}
static void Text(BinaryWriter writer, string value)
{
    byte[] bytes = Encoding.UTF8.GetBytes(value);
    writer.Write((byte)'X'); writer.Write(bytes.Length); writer.Write(bytes);
}
static void Global(BinaryWriter writer, string module, string name)
{
    writer.Write((byte)'c'); writer.Write(Encoding.ASCII.GetBytes(module + "\n" + name + "\n"));
}
