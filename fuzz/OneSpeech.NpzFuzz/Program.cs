using System.IO.Compression;
using System.Text;
using OneSpeech.Formats;
using SharpFuzz;

if (args is ["seed", var directory])
{
    Directory.CreateDirectory(directory);
    byte[] npy = Npy("<f8", "(2, 2)", [1.0, 2.0, 3.0, 4.0]);
    File.WriteAllBytes(Path.Combine(directory, "array.npy"), npy);
    using var zip = new MemoryStream();
    using (var archive = new ZipArchive(zip, ZipArchiveMode.Create, leaveOpen: true))
    {
        using var entry = archive.CreateEntry("mu.npy").Open();
        entry.Write(Npy("<f4", "(3,)", [1.0, 2.0, 3.0]));
    }
    File.WriteAllBytes(Path.Combine(directory, "archive.npz"), zip.ToArray());
    return;
}

Fuzzer.LibFuzzer.Run(data =>
{
    try
    {
        using var stream = new MemoryStream(data.ToArray());
        _ = NpzArchive.Read(stream);
    }
    catch (InvalidDataException) { }
    try { _ = NpzArchive.ParseNpy(data); }
    catch (InvalidDataException) { }
});

static byte[] Npy(string descr, string shape, double[] values)
{
    string header = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}";
    header = header.PadRight(header.Length + (64 - (10 + header.Length + 1) % 64) % 64) + "\n";
    using var buffer = new MemoryStream();
    buffer.Write([0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0, (byte)(header.Length & 255), (byte)(header.Length >> 8)]);
    buffer.Write(Encoding.ASCII.GetBytes(header));
    foreach (double v in values) buffer.Write(descr == "<f8" ? BitConverter.GetBytes(v) : BitConverter.GetBytes((float)v));
    return buffer.ToArray();
}
