using System.Buffers.Binary;
using System.Collections.Immutable;
using System.Text;
using System.Text.Json;

namespace OneSpeech.Formats;

// Reads tensor data only. No code objects, pickle, or remote-code loading are supported.
// Limits are independent of claimed file sizes: header <=8 MiB, names <=1024 UTF-8 bytes,
// <=50,000 tensors/metadata keys, rank <=16, and <=512 MiB per materialized float array.
internal sealed class SafeTensorFile : IDisposable
{
    private const int MaxHeaderBytes = 8 * 1024 * 1024;
    private const int MaxEntries = 50000;
    private const long MaxFloatElements = 128 * 1024 * 1024;
    private readonly Stream stream;
    private readonly bool leaveOpen;
    private readonly long dataStart;
    private readonly Dictionary<string, SafeTensorInfo> tensors = new(StringComparer.Ordinal);

    internal SafeTensorFile(Stream stream, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead || !stream.CanSeek) throw new ArgumentException("Safetensors requires a readable, seekable stream.", nameof(stream));
        this.stream = stream;
        this.leaveOpen = leaveOpen;
        try
        {
            stream.Position = 0;
            if (stream.Length < 10) throw Invalid("Truncated length/header.");
            byte[] prefix = new byte[8];
            stream.ReadExactly(prefix);
            ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
            if (headerSize is < 2 or > MaxHeaderBytes || headerSize > (ulong)(stream.Length - 8)) throw Invalid("Header length outside bounds.");
            byte[] header = new byte[(int)headerSize];
            stream.ReadExactly(header);
            if (header[0] != (byte)'{') throw Invalid("Header must begin with an object.");
            _ = new UTF8Encoding(false, true).GetCharCount(header);
            dataStart = 8 + (long)headerSize;
            Parse(header);
            ValidateRanges(stream.Length - dataStart);
        }
        catch (Exception ex) when (ex is JsonException or DecoderFallbackException or OverflowException or EndOfStreamException)
        {
            if (!leaveOpen) stream.Dispose();
            throw Invalid("Malformed safetensors header.", ex);
        }
        catch
        {
            if (!leaveOpen) stream.Dispose();
            throw;
        }
    }

    internal IReadOnlyDictionary<string, SafeTensorInfo> Tensors => tensors;

    internal void CopyTensorTo(string name, Stream destination, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(destination);
        SafeTensorInfo tensor = tensors[name];
        stream.Position = dataStart + tensor.Offset;
        byte[] buffer = new byte[65536];
        long remaining = tensor.ByteLength;
        while (remaining > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = (int)Math.Min(remaining, buffer.Length);
            stream.ReadExactly(buffer.AsSpan(0, count));
            destination.Write(buffer.AsSpan(0, count));
            remaining -= count;
        }
    }

    internal float[] ReadFloat32(string name, CancellationToken cancellationToken)
    {
        SafeTensorInfo tensor = tensors[name];
        if (tensor.DataType is not ("F32" or "F16" or "BF16")) throw new NotSupportedException("Expected F32, F16 or BF16 tensor.");
        if (tensor.ElementCount > MaxFloatElements) throw Invalid("Float tensor exceeds the 512 MiB materialization limit.");
        cancellationToken.ThrowIfCancellationRequested();
        float[] result = new float[(int)tensor.ElementCount];
        byte[] buffer = new byte[65536];
        int elementSize = tensor.DataType == "F32" ? 4 : 2;
        stream.Position = dataStart + tensor.Offset;
        for (int offset = 0; offset < result.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int count = Math.Min(result.Length - offset, buffer.Length / elementSize);
            stream.ReadExactly(buffer.AsSpan(0, count * elementSize));
            for (int i = 0; i < count; i++)
            {
                ReadOnlySpan<byte> bytes = buffer.AsSpan(i * elementSize, elementSize);
                result[offset + i] = tensor.DataType switch
                {
                    "F32" => BinaryPrimitives.ReadSingleLittleEndian(bytes),
                    "F16" => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes)),
                    _ => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(bytes) << 16)
                };
            }
            offset += count;
        }
        return result;
    }

    private void Parse(ReadOnlySpan<byte> header)
    {
        Utf8JsonReader reader = new(header, new JsonReaderOptions { MaxDepth = 8 });
        RequireRead(ref reader, JsonTokenType.StartObject);
        HashSet<string> keys = new(StringComparer.Ordinal);
        while (Read(ref reader) != JsonTokenType.EndObject)
        {
            string name = ReadString(ref reader, JsonTokenType.PropertyName);
            if (keys.Count >= MaxEntries || !keys.Add(name)) throw Invalid("Too many or duplicate header entries.");
            RequireRead(ref reader, JsonTokenType.StartObject);
            if (name == "__metadata__") { ReadMetadata(ref reader); continue; }
            string? dtype = null;
            ImmutableArray<long> shape = default;
            long begin = -1, end = -1;
            int fields = 0;
            while (Read(ref reader) != JsonTokenType.EndObject)
            {
                string field = ReadString(ref reader, JsonTokenType.PropertyName);
                int flag = field switch { "dtype" => 1, "shape" => 2, "data_offsets" => 4, _ => throw Invalid("Unknown tensor field.") };
                if ((fields & flag) != 0) throw Invalid("Duplicate tensor field.");
                fields |= flag;
                if (flag == 1)
                {
                    Read(ref reader);
                    dtype = ReadString(ref reader, JsonTokenType.String);
                }
                else if (flag == 2)
                {
                    RequireRead(ref reader, JsonTokenType.StartArray);
                    var dimensions = ImmutableArray.CreateBuilder<long>();
                    while (Read(ref reader) != JsonTokenType.EndArray)
                    {
                        if (dimensions.Count >= 16) throw Invalid("Tensor rank exceeds 16.");
                        dimensions.Add(ReadNonnegativeInteger(ref reader));
                    }
                    shape = dimensions.ToImmutable();
                }
                else
                {
                    RequireRead(ref reader, JsonTokenType.StartArray);
                    Read(ref reader); begin = ReadNonnegativeInteger(ref reader);
                    Read(ref reader); end = ReadNonnegativeInteger(ref reader);
                    RequireRead(ref reader, JsonTokenType.EndArray);
                }
            }
            if (fields != 7 || dtype is null || end < begin) throw Invalid("Incomplete tensor or invalid offsets.");
            int size = dtype switch
            {
                "BOOL" or "U8" or "I8" => 1,
                "F16" or "BF16" or "I16" or "U16" => 2,
                "F32" or "I32" or "U32" => 4,
                "F64" or "I64" or "U64" => 8,
                _ => throw new NotSupportedException("Unsupported safetensors dtype: " + dtype)
            };
            long elements = shape.Contains(0) ? 0 : 1;
            if (elements != 0) foreach (long dimension in shape) elements = checked(elements * dimension);
            if (checked(elements * size) != end - begin) throw Invalid("Shape, dtype and byte range disagree.");
            tensors.Add(name, new(dtype, shape, begin, end - begin, elements));
        }
        if (reader.Read()) throw Invalid("Trailing JSON value.");
    }

    private static void ReadMetadata(ref Utf8JsonReader reader)
    {
        HashSet<string> keys = new(StringComparer.Ordinal);
        while (Read(ref reader) != JsonTokenType.EndObject)
        {
            string key = ReadString(ref reader, JsonTokenType.PropertyName);
            if (keys.Count >= MaxEntries || !keys.Add(key)) throw Invalid("Too many or duplicate metadata keys.");
            RequireRead(ref reader, JsonTokenType.String);
            _ = ReadString(ref reader, JsonTokenType.String, 65536);
        }
    }

    private void ValidateRanges(long dataLength)
    {
        long next = 0;
        foreach (SafeTensorInfo tensor in tensors.Values.OrderBy(t => t.Offset).ThenBy(t => t.ByteLength))
        {
            if (tensor.Offset != next || tensor.ByteLength > dataLength - next) throw Invalid("Tensor ranges overlap, contain a hole, or exceed the file.");
            next += tensor.ByteLength;
        }
        if (next != dataLength) throw Invalid("Unindexed data at end of file.");
    }

    private static string ReadString(ref Utf8JsonReader reader, JsonTokenType type, int maxBytes = 1024)
    {
        if (reader.TokenType != type || reader.ValueSpan.Length > maxBytes) throw Invalid("Expected string of at most 1024 UTF-8 bytes.");
        try { return reader.GetString()!; }
        catch (InvalidOperationException ex) { throw Invalid("Invalid Unicode string.", ex); }
    }
    private static long ReadNonnegativeInteger(ref Utf8JsonReader reader)
    {
        if (reader.TokenType != JsonTokenType.Number || !reader.TryGetInt64(out long value) || value < 0) throw Invalid("Expected nonnegative Int64.");
        return value;
    }
    private static JsonTokenType Read(ref Utf8JsonReader reader)
    {
        if (!reader.Read()) throw Invalid("Truncated header.");
        return reader.TokenType;
    }
    private static void RequireRead(ref Utf8JsonReader reader, JsonTokenType type)
    {
        if (Read(ref reader) != type) throw Invalid("Unexpected JSON token.");
    }
    private static InvalidDataException Invalid(string message, Exception? inner = null) => new(message, inner);
    public void Dispose() { if (!leaveOpen) stream.Dispose(); }
}

internal sealed record SafeTensorInfo(string DataType, ImmutableArray<long> Shape, long Offset, long ByteLength, long ElementCount);
