using System.Buffers.Binary;
using System.Text;
using OneSpeech.Formats;
using SharpFuzz;

if (args.Length == 2 && args[0] == "--seed")
{
    Directory.CreateDirectory(args[1]);
    foreach ((string name, string header, int length) in new[]
    {
        ("empty", "{}", 0),
        ("scalar", """{"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}""", 4),
        ("vector", """{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":"pt"}}""", 4),
        ("zero", """{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}""", 0)
    })
    {
        byte[] json = Encoding.UTF8.GetBytes(header);
        byte[] file = new byte[8 + json.Length + length];
        BinaryPrimitives.WriteUInt64LittleEndian(file, (ulong)json.Length);
        json.CopyTo(file, 8);
        File.WriteAllBytes(Path.Combine(args[1], name), file);
    }
    return;
}

Fuzzer.LibFuzzer.Run(data =>
{
    try
    {
        using var stream = new MemoryStream(data.ToArray(), writable: false);
        using var file = new SafeTensorFile(stream);
        foreach (var pair in file.Tensors)
            if (pair.Value.ElementCount <= 16384 && pair.Value.DataType is "F32" or "F16" or "BF16")
            {
                _ = file.ReadFloat32(pair.Key, CancellationToken.None);
                using var expanded = new MemoryStream();
                file.CopyFloat32To(pair.Key, expanded, CancellationToken.None);
            }
    }
    catch (InvalidDataException) { }
    catch (NotSupportedException) { }
});
