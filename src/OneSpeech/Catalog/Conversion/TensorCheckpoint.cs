using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

// Reads a deliberately restricted data language, not Python objects. Globals are inert names;
// reductions only construct the tensor/ordered-map records defined below. Nothing is invoked.
internal sealed class TensorCheckpoint
{
    internal sealed record Tensor(int[] Shape, float[] Values);
    private sealed record Global(string Module, string Name);
    private sealed record Storage(string Key, string Type, long Count);
    private sealed record View(Storage Storage, long Offset, long[] Shape, long[] Strides);
    private static readonly object Mark = new();
    private static readonly UTF8Encoding Utf8 = new(false, true);
    private const int MaxItems = 200000;
    private const long MaxBytes = 256L * 1024 * 1024;

    internal static Dictionary<string, Tensor> Read(Stream stream, bool largeModel = false, bool segmentation = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        try { return ReadCore(stream, largeModel, segmentation); }
        catch (OverflowException e) { throw new InvalidDataException("Checkpoint integer overflow.", e); }
        catch (DecoderFallbackException e) { throw new InvalidDataException("Invalid checkpoint UTF-8.", e); }
        catch (EndOfStreamException e) { throw new InvalidDataException("Truncated checkpoint entry.", e); }
    }

    private static Dictionary<string, Tensor> ReadCore(Stream stream, bool largeModel, bool segmentation)
    {
        long maxBytes = largeModel ? 1536L * 1024 * 1024 : MaxBytes;
        if (!stream.CanSeek || stream.Length > maxBytes || stream.Length < 22) throw Bad("Checkpoint size or stream.");
        CheckDirectory(stream);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
        if (archive.Entries.Count > 20000) throw Bad("Too many ZIP entries.");
        var entries = new Dictionary<string, ZipArchiveEntry>(StringComparer.Ordinal);
        long total = 0;
        foreach (var entry in archive.Entries)
        {
            total = checked(total + entry.Length);
            if (entry.FullName.Length > 1024 || entry.Length > maxBytes || total > maxBytes || !entries.TryAdd(entry.FullName, entry)) throw Bad("ZIP bounds or duplicate.");
        }
        var indices = archive.Entries.Where(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal)).ToArray();
        if (indices.Length != 1) throw Bad("Expected one tensor index.");
        var pickle = indices[0];
        string prefix = pickle.FullName[..^8];
        var root = new Parser(Bytes(pickle, 8 * 1024 * 1024), segmentation).Run();
        if (entries.TryGetValue(prefix + "byteorder", out var byteOrder) && Utf8.GetString(Bytes(byteOrder, 16)) != "little") throw Bad("Only little-endian storage is supported.");
        if (root is not Dictionary<string, object?> map) throw Bad("Expected state dictionary.");
        if (segmentation)
            map = map.TryGetValue("state_dict", out var state) && state is Dictionary<string, object?> dictionary ? dictionary : throw Bad("Missing segmentation state dictionary.");
        if (largeModel && map.TryGetValue("model", out var nested))
            map = nested as Dictionary<string, object?> ?? throw Bad("Invalid model tensor dictionary.");
        var storages = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        long outputBytes = 0;
        foreach (var pair in map)
        {
            if (pair.Value is not View view) throw Bad("Only named tensor values are supported.");
            int width = view.Storage.Type switch { "FloatStorage" => 4, "LongStorage" => 8, "DoubleStorage" => 8, "HalfStorage" => 2, _ => throw Bad("Unsupported storage type.") };
            if (view.Storage.Count < 0 || view.Storage.Count > maxBytes / width) throw Bad("Storage bounds.");
            long storageLength = view.Storage.Count * width;
            if (!storages.TryGetValue(view.Storage.Key, out var raw))
            {
                if (!entries.TryGetValue(prefix + "data/" + view.Storage.Key, out var entry) || entry.Length != storageLength) throw Bad("Storage length mismatch.");
                raw = Bytes(entry, (int)maxBytes);
                storages.Add(view.Storage.Key, raw);
            }
            if (raw.LongLength != storageLength || view.Shape.Length != view.Strides.Length || view.Shape.Length > 8 || view.Offset < 0) throw Bad("Invalid tensor view.");
            long count = 1, maximum = view.Offset;
            var shape = new int[view.Shape.Length];
            for (int i = 0; i < shape.Length; i++)
            {
                if (view.Shape[i] < 0 || view.Shape[i] > 10000000 || view.Strides[i] < 0 || view.Strides[i] > maxBytes) throw Bad("Tensor dimension bounds.");
                count = checked(count * view.Shape[i]);
                maximum = checked(maximum + Math.Max(0, view.Shape[i] - 1) * view.Strides[i]);
                shape[i] = (int)view.Shape[i];
                if (count > maxBytes / 4) throw Bad("Tensor allocation bound.");
            }
            if ((count > 0 && maximum >= view.Storage.Count) || (count == 0 && view.Offset > view.Storage.Count)) throw Bad("Tensor outside storage.");
            outputBytes = checked(outputBytes + count * 4);
            if (outputBytes > maxBytes) throw Bad("Total tensor allocation bound.");
            var values = new float[(int)count];
            for (int i = 0; i < values.Length; i++)
            {
                long index = view.Offset;
                int remaining = i;
                for (int d = shape.Length - 1; d >= 0; d--)
                {
                    index += (remaining % shape[d]) * view.Strides[d];
                    remaining /= shape[d];
                }
                int offset = checked((int)(index * width));
                values[i] = view.Storage.Type switch
                {
                    "FloatStorage" => BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(offset, 4)),
                    "LongStorage" => BinaryPrimitives.ReadInt64LittleEndian(raw.AsSpan(offset, 8)),
                    "DoubleStorage" => (float)BinaryPrimitives.ReadDoubleLittleEndian(raw.AsSpan(offset, 8)),
                    "HalfStorage" => (float)BitConverter.Int16BitsToHalf(BinaryPrimitives.ReadInt16LittleEndian(raw.AsSpan(offset, 2))),
                    _ => throw Bad("Unsupported storage.")
                };
            }
            result.Add(pair.Key, new Tensor(shape, values));
        }
        return result;
    }

    // Bound the central directory before ZipArchive materializes its entry objects. The checkpoints
    // use ordinary 32-bit directory offsets; ZIP64-only and multi-disk archives are deliberately refused.
    private static void CheckDirectory(Stream stream)
    {
        int length = (int)Math.Min(stream.Length, 65557);
        byte[] tail = new byte[length];
        stream.Position = stream.Length - length;
        stream.ReadExactly(tail);
        for (int i = length - 22; i >= 0; i--)
        {
            var record = tail.AsSpan(i);
            if (BinaryPrimitives.ReadUInt32LittleEndian(record) != 0x06054b50 ||
                i + 22 + BinaryPrimitives.ReadUInt16LittleEndian(record[20..]) != length) continue;
            int count = BinaryPrimitives.ReadUInt16LittleEndian(record[10..]);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(record[12..]);
            uint offset = BinaryPrimitives.ReadUInt32LittleEndian(record[16..]);
            if (BinaryPrimitives.ReadUInt16LittleEndian(record[4..]) != 0 ||
                BinaryPrimitives.ReadUInt16LittleEndian(record[6..]) != 0 ||
                BinaryPrimitives.ReadUInt16LittleEndian(record[8..]) != count ||
                count > 20000 || size > 8 * 1024 * 1024 ||
                (long)offset + size > stream.Length - length + i)
                throw Bad("ZIP central directory bound.");
            stream.Position = 0;
            return;
        }
        throw Bad("Missing ZIP end record.");
    }

    private static byte[] Bytes(ZipArchiveEntry entry, int limit)
    {
        if (entry.Length < 0 || entry.Length > limit) throw Bad("ZIP entry allocation bound.");
        var bytes = new byte[(int)entry.Length];
        using var input = entry.Open();
        input.ReadExactly(bytes);
        if (input.ReadByte() != -1) throw Bad("ZIP entry length mismatch.");
        return bytes;
    }

    private static InvalidDataException Bad(string message) => new(message);

    // Checkpoint descriptors are inert data: their class names are allowlisted, but
    // no constructors, enum functions, attributes, or code objects are evaluated.
    private sealed record Metadata(Global Type, object?[] Arguments);

    private sealed class Parser(byte[] data, bool segmentation)
    {
        private readonly List<object?> stack = [];
        private readonly Dictionary<int, object?> memo = [];
        private int position;
        private int operations;
        private int allocatedItems;

        internal object? Run()
        {
            while (position < data.Length)
            {
                if (++operations > 2000000 || stack.Count > MaxItems || memo.Count > MaxItems) throw Bad("Pickle complexity bound.");
                int op = Byte();
                switch (op)
                {
                    case 0x80: if (Byte() is < 2 or > 5) throw Bad("Unsupported pickle protocol."); break;
                    case 0x95: { long size = Int64(); if (size < 0 || size > data.Length - position) throw Bad("Invalid frame."); break; }
                    case (int)'.': if (position != data.Length || stack.Count != 1) throw Bad("Invalid pickle stop."); return Pop();
                    case (int)'(': stack.Add(Mark); break;
                    case (int)')': stack.Add(System.Array.Empty<object?>()); break;
                    case (int)'}': stack.Add(new Dictionary<string, object?>(StringComparer.Ordinal)); break;
                    case (int)']': stack.Add(new List<object?>()); break;
                    case (int)'N': stack.Add(null); break;
                    case 0x88: stack.Add(true); break;
                    case 0x89: stack.Add(false); break;
                    case (int)'K': stack.Add((long)Byte()); break;
                    case (int)'M': stack.Add((long)UInt16()); break;
                    case (int)'J': stack.Add((long)Int32()); break;
                    case (int)'G': stack.Add(BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64BigEndian(Take(8)))); break;
                    case 0x81:
                    {
                        var arguments = Array(Pop());
                        if (!segmentation || Pop() is not Global type || !IsMetadata(type.Module, type.Name)) throw Bad("Unsupported object data.");
                        stack.Add(new Metadata(type, arguments)); break;
                    }
                    case 0x8a:
                    {
                        int n = Byte();
                        if (n is < 1 or > 8) throw Bad("Integer out of range.");
                        var bytes = Take(n);
                        long value = (bytes[n - 1] & 128) != 0 ? -1 : 0;
                        for (int i = n - 1; i >= 0; i--) value = (value << 8) | bytes[i];
                        stack.Add(value);
                        break;
                    }
                    case (int)'X': stack.Add(Text(Int32())); break;
                    case 0x8c: stack.Add(Text(Byte())); break;
                    case (int)'c': stack.Add(AllowedGlobal(Line(), Line())); break;
                    case 0x93:
                    {
                        string name = String(Pop()), module = String(Pop());
                        stack.Add(AllowedGlobal(module, name)); break;
                    }
                    case (int)'q': Memo(Byte()); break;
                    case (int)'r': Memo(Int32()); break;
                    case 0x94: Memo(memo.Count); break;
                    case (int)'h': Get(Byte()); break;
                    case (int)'j': Get(Int32()); break;
                    case 0x85: Tuple(1); break;
                    case 0x86: Tuple(2); break;
                    case 0x87: Tuple(3); break;
                    case (int)'t': stack.Add(Items()); break;
                    case (int)'u': SetItems(Items()); break;
                    case (int)'s': { object? value = Pop(); object? key = Pop(); SetItems([key, value]); break; }
                    case (int)'e': { var values = Items(); Charge(values.Length); if (Peek() is not List<object?> list || list.Count + values.Length > MaxItems) throw Bad("Invalid list."); list.AddRange(values); break; }
                    case (int)'a': { Charge(1); var value = Pop(); if (Peek() is not List<object?> list || list.Count >= MaxItems) throw Bad("Invalid list."); list.Add(value); break; }
                    case (int)'Q':
                    {
                        var tuple = Array(Pop());
                        if (tuple.Length != 5 || String(tuple[0]) != "storage" || tuple[1] is not Global g || g.Module != "torch") throw Bad("Invalid persistent storage.");
                        string key = String(tuple[2]);
                        if (key.Length is < 1 or > 128 || key.Any(c => !char.IsAsciiDigit(c))) throw Bad("Invalid storage key.");
                        _ = String(tuple[3]); // Device labels are data; no device is opened.
                        stack.Add(new Storage(key, g.Name, Number(tuple[4]))); break;
                    }
                    case (int)'R':
                    {
                        var args = Array(Pop());
                        if (Pop() is not Global callable) throw Bad("Invalid inert reduction.");
                        if (callable == new Global("collections", "OrderedDict") && args.Length == 0)
                            stack.Add(new Dictionary<string, object?>(StringComparer.Ordinal));
                        else if (callable == new Global("torch._utils", "_rebuild_tensor_v2") && args.Length == 6 && args[0] is Storage storage && args[4] is bool && args[5] is Dictionary<string, object?> hooks && hooks.Count == 0)
                        {
                            var shape = Array(args[2]); var strides = Array(args[3]);
                            if (shape.Length > 8 || strides.Length != shape.Length) throw Bad("Tensor rank bound.");
                            Charge(shape.Length * 2);
                            stack.Add(new View(storage, Number(args[1]), shape.Select(Number).ToArray(), strides.Select(Number).ToArray()));
                        }
                        else if (segmentation && IsMetadata(callable.Module, callable.Name)) stack.Add(new Metadata(callable, args));
                        else throw Bad("Unsupported reduction data.");
                        break;
                    }
                    case (int)'b':
                    {
                        // OrderedDict's optional _metadata is discarded without interpreting it.
                        var objectState = Pop();
                        if (segmentation && Peek() is Metadata && objectState is Dictionary<string, object?>) break;
                        if (objectState is not Dictionary<string, object?> state || state.Count != 1 || !state.TryGetValue("_metadata", out var metadata) || metadata is not Dictionary<string, object?> || Peek() is not Dictionary<string, object?>)
                            throw Bad("Unsupported pickle state.");
                        break;
                    }
                    default: throw Bad($"Unsupported pickle opcode {op:X2} at {position - 1}.");
                }
            }
            throw Bad("Truncated pickle.");
        }

        private static bool IsMetadata(string module, string name) => (module == "pyannote.audio.core.task" && name is "Specifications" or "Problem" or "Resolution") || (module == "torch.torch_version" && name == "TorchVersion");

        private Global AllowedGlobal(string module, string name)
        {
            if ((module == "collections" && name == "OrderedDict") || (module == "torch._utils" && name == "_rebuild_tensor_v2") ||
                (module == "torch" && name is "FloatStorage" or "LongStorage" or "HalfStorage" or "DoubleStorage")) return new(module, name);
            if (segmentation && IsMetadata(module, name)) return new(module, name);
            throw Bad("Global outside inert tensor allowlist.");
        }
        private void SetItems(object?[] items)
        {
            if (Peek() is not Dictionary<string, object?> map || items.Length % 2 != 0 || map.Count + items.Length / 2 > MaxItems) throw Bad("Invalid dictionary.");
            Charge(items.Length);
            for (int i = 0; i < items.Length; i += 2)
                if (!map.TryAdd(String(items[i]), items[i + 1])) throw Bad("Duplicate dictionary key.");
        }
        private void Tuple(int count)
        {
            if (stack.Count < count) throw Bad("Pickle stack underflow.");
            Charge(count);
            var tuple = stack.GetRange(stack.Count - count, count).ToArray();
            stack.RemoveRange(stack.Count - count, count); stack.Add(tuple);
        }
        private object?[] Items()
        {
            int start = stack.LastIndexOf(Mark);
            if (start < 0) throw Bad("Missing pickle mark.");
            Charge(stack.Count - start - 1);
            var result = stack.GetRange(start + 1, stack.Count - start - 1).ToArray();
            stack.RemoveRange(start, stack.Count - start); return result;
        }
        private void Charge(int count)
        {
            allocatedItems = checked(allocatedItems + count);
            if (allocatedItems > 2000000) throw Bad("Aggregate pickle allocation bound.");
        }
        private void Memo(int key) { if (key < 0 || key > MaxItems || !memo.TryAdd(key, Peek())) throw Bad("Invalid memo key."); }
        private void Get(int key) { if (!memo.TryGetValue(key, out var value)) throw Bad("Missing memo key."); stack.Add(value); }
        private object? Peek() => stack.Count == 0 ? throw Bad("Pickle stack underflow.") : stack[^1];
        private object? Pop() { var value = Peek(); stack.RemoveAt(stack.Count - 1); return value; }
        private static string String(object? value) => value as string ?? throw Bad("Expected string.");
        private static long Number(object? value) => value is long n ? n : throw Bad("Expected integer.");
        private static object?[] Array(object? value) => value as object?[] ?? throw Bad("Expected tuple.");
        private byte Byte() => Take(1)[0];
        private ushort UInt16() => BinaryPrimitives.ReadUInt16LittleEndian(Take(2));
        private int Int32() => BinaryPrimitives.ReadInt32LittleEndian(Take(4));
        private long Int64() => BinaryPrimitives.ReadInt64LittleEndian(Take(8));
        private string Text(int length) { if (length < 0 || length > 65536) throw Bad("String bound."); return Utf8.GetString(Take(length)); }
        private string Line()
        {
            int end = System.Array.IndexOf(data, (byte)'\n', position);
            if (end < position || end - position > 1024) throw Bad("Invalid global name.");
            string text = Text(end - position); position++; return text;
        }
        private ReadOnlySpan<byte> Take(int count)
        {
            if (count < 0 || count > data.Length - position) throw Bad("Truncated pickle operand.");
            var result = data.AsSpan(position, count); position += count; return result;
        }
    }
}
