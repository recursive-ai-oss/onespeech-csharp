using System.Globalization;
using System.IO.Compression;
using System.Text;

namespace OneSpeech.Formats;

/// <summary>One array read from an NPY payload: C-order values widened to double.</summary>
internal sealed record NpyArray(int[] Shape, double[] Values);

/// <summary>Bounded reader for the NumPy NPY/NPZ array containers (NumPy "npy-format" specification, versions 1.0-3.0).
/// Reads little-endian float32/float64/int32/int64 and one-byte bool/uint8 arrays of at most four dimensions in C order; anything else, any
/// size beyond the limits, or a malformed header throws <see cref="InvalidDataException"/>. Nothing is executed.
/// An NPZ file is a ZIP archive whose members are "&lt;key&gt;.npy"; compressed members are inflated within the byte limits.</summary>
internal static class NpzArchive
{
    internal const int MaxEntries = 64;
    internal const long MaxArrayBytes = 64L * 1024 * 1024;
    internal const long MaxFileBytes = 256L * 1024 * 1024;
    private const int MaxHeader = 4096;
    private static ReadOnlySpan<byte> Magic => [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    internal static Dictionary<string, NpyArray> Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanSeek || stream.Length > MaxFileBytes || stream.Length < 22) throw Bad("NPZ size or stream.");
        try
        {
            using var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
            if (archive.Entries.Count > MaxEntries) throw Bad("Too many NPZ members.");
            var result = new Dictionary<string, NpyArray>(StringComparer.Ordinal);
            long total = 0;
            foreach (var entry in archive.Entries)
            {
                if (!entry.FullName.EndsWith(".npy", StringComparison.Ordinal) || entry.FullName.Length > 260) throw Bad("Unexpected NPZ member name.");
                total = checked(total + entry.Length);
                if (entry.Length > MaxArrayBytes + MaxHeader || total > MaxFileBytes) throw Bad("NPZ member too large.");
                byte[] bytes = new byte[entry.Length];
                using (var input = entry.Open())
                {
                    int read = 0;
                    while (read < bytes.Length)
                    {
                        int n = input.Read(bytes, read, bytes.Length - read);
                        if (n == 0) throw Bad("Truncated NPZ member.");
                        read += n;
                    }
                    if (input.ReadByte() >= 0) throw Bad("NPZ member longer than declared.");
                }
                if (!result.TryAdd(entry.FullName[..^4], ParseNpy(bytes))) throw Bad("Duplicate NPZ member.");
            }
            return result;
        }
        catch (InvalidDataException) { throw; }
        catch (OverflowException e) { throw new InvalidDataException("NPZ arithmetic overflow.", e); }
    }

    /// <summary>Parses one complete NPY file image.</summary>
    internal static NpyArray ParseNpy(ReadOnlySpan<byte> data)
    {
        if (data.Length < 10 || !data[..6].SequenceEqual(Magic)) throw Bad("Missing NPY magic.");
        int major = data[6];
        int headerLength, headerStart;
        if (major == 1) { headerLength = data[8] | data[9] << 8; headerStart = 10; }
        else if (major is 2 or 3)
        {
            if (data.Length < 12) throw Bad("Truncated NPY header.");
            uint len = (uint)(data[8] | data[9] << 8 | data[10] << 16 | data[11] << 24);
            if (len > MaxHeader) throw Bad("NPY header too large.");
            headerLength = (int)len; headerStart = 12;
        }
        else throw Bad("Unsupported NPY version.");
        if (headerLength > MaxHeader || headerStart + headerLength > data.Length) throw Bad("NPY header bounds.");
        string header;
        try { header = new UTF8Encoding(false, true).GetString(data.Slice(headerStart, headerLength)); }
        catch (DecoderFallbackException e) { throw new InvalidDataException("NPY header is not UTF-8.", e); }
        string descr = Field(header, "'descr'"), order = Field(header, "'fortran_order'"), shapeText = Field(header, "'shape'");
        if (order != "False") throw Bad("Only C-order arrays are supported.");
        int width = descr switch { "'<f8'" => 8, "'<f4'" => 4, "'<i8'" => 8, "'<i4'" => 4, "'|b1'" or "'|u1'" => 1, _ => throw Bad("Unsupported NPY dtype.") };
        if (!shapeText.StartsWith('(') || !shapeText.EndsWith(')')) throw Bad("Bad NPY shape.");
        var dims = new List<int>();
        long count = 1;
        foreach (string part in shapeText[1..^1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (!int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out int d) || dims.Count == 4) throw Bad("Bad NPY dimension.");
            dims.Add(d);
            count = checked(count * d);
            if (count > MaxArrayBytes / width) throw Bad("NPY array too large.");
        }
        var payload = data[(headerStart + headerLength)..];
        if (payload.Length != count * width) throw Bad("NPY payload length does not match its shape.");
        var values = new double[count];
        for (int i = 0; i < values.Length; i++)
        {
            var item = payload.Slice(i * width, width);
            values[i] = descr switch
            {
                "'<f8'" => BitConverter.ToDouble(item),
                "'<f4'" => BitConverter.ToSingle(item),
                "'<i8'" => BitConverter.ToInt64(item),
                "'|b1'" or "'|u1'" => item[0],
                _ => BitConverter.ToInt32(item),
            };
        }
        return new NpyArray([.. dims], values);
    }

    // Header is a Python dict literal with three known keys; values never contain nested braces.
    private static string Field(string header, string key)
    {
        int at = header.IndexOf(key, StringComparison.Ordinal);
        if (at < 0) throw Bad("NPY header lacks " + key + ".");
        int colon = header.IndexOf(':', at + key.Length);
        if (colon < 0) throw Bad("Bad NPY header.");
        int start = colon + 1;
        while (start < header.Length && header[start] == ' ') start++;
        int end;
        if (start < header.Length && header[start] == '(') { end = header.IndexOf(')', start); if (end < 0) throw Bad("Bad NPY shape."); end++; }
        else
        {
            end = start;
            while (end < header.Length && header[end] != ',' && header[end] != '}') end++;
        }
        return header[start..end].Trim();
    }

    private static InvalidDataException Bad(string message) => new(message);
}
