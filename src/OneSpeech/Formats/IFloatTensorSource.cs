using System.Buffers.Binary;

namespace OneSpeech.Formats;

// Graph construction depends on tensor data, not a checkpoint container or executable model code.
internal interface IFloatTensorSource
{
    long[] GetShape(string name);
    void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken);
}

internal sealed class SafeTensorSource(SafeTensorFile file) : IFloatTensorSource
{
    public long[] GetShape(string name) => file.Tensors[name].Shape.ToArray();

    public void CopyFloat32To(string name, Stream destination, CancellationToken cancellationToken)
    {
        if (file.Tensors[name].DataType == "F32") { file.CopyTensorTo(name, destination, cancellationToken); return; }
        float[] values = file.ReadFloat32(name, cancellationToken);
        byte[] bytes = new byte[65536];
        for (int offset = 0; offset < values.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(values.Length - offset, bytes.Length / 4);
            for (int i = 0; i < count; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[offset + i]);
            destination.Write(bytes, 0, count * 4);
            offset += count;
        }
    }
}
