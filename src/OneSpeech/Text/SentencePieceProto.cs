using System.Buffers.Binary;
using System.Text;

namespace OneSpeech.Text;

/// <summary>What a SentencePiece <c>.model</c> file holds that a tokenizer needs.</summary>
internal sealed class SentencePieceData
{
    /// <summary>The piece types of the published schema (<c>ModelProto.SentencePiece.Type</c>).</summary>
    internal const byte Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6;

    /// <summary>The pieces, by ID.</summary>
    internal string[] Pieces { get; init; } = [];

    /// <summary>The score (log probability) of each piece.</summary>
    internal float[] Scores { get; init; } = [];

    /// <summary>The type of each piece.</summary>
    internal byte[] Types { get; init; } = [];

    /// <summary>The trainer's model type: 1 unigram, 2 BPE, 3 word, 4 character.</summary>
    internal int ModelType { get; init; } = 1;

    /// <summary>The ID of the unknown piece.</summary>
    internal int UnkId { get; init; }

    /// <summary>Whether unknown characters are spelled with byte pieces.</summary>
    internal bool ByteFallback { get; init; }

    /// <summary>The normalizer rule set name, for example <c>nmt_nfkc</c> or <c>identity</c>.</summary>
    internal string NormalizerName { get; init; } = string.Empty;

    /// <summary>The compiled normalization table, empty when the model has none.</summary>
    internal byte[] Charsmap { get; init; } = [];

    /// <summary>Whether a space is put in front of the text.</summary>
    internal bool AddDummyPrefix { get; init; } = true;

    /// <summary>Whether leading and trailing spaces are removed and runs of spaces collapsed.</summary>
    internal bool RemoveExtraWhitespaces { get; init; } = true;

    /// <summary>Whether a space is written as U+2581 in pieces.</summary>
    internal bool EscapeWhitespaces { get; init; } = true;
}

/// <summary>
/// A bounded reader for the protocol-buffer message of a SentencePiece model, following the published schema
/// (<c>sentencepiece_model.proto</c>): <c>ModelProto</c> field 1 (pieces: piece 1, score 2, type 3), 2 (trainer spec: model_type 3,
/// byte_fallback 35, unk_id 40) and 3 (normalizer spec: name 1, precompiled_charsmap 2, add_dummy_prefix 3,
/// remove_extra_whitespaces 4, escape_whitespaces 5). Other fields are skipped. Every length is checked against the data before
/// anything is allocated.
/// </summary>
internal static class SentencePieceProto
{
    /// <summary>The largest file read, in bytes.</summary>
    internal const int MaxFileBytes = 4 << 20;

    /// <summary>The most pieces accepted.</summary>
    internal const int MaxPieces = 100_000;

    /// <summary>The longest piece accepted, in UTF-8 bytes.</summary>
    internal const int MaxPieceBytes = 4096;

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    internal static SentencePieceData ReadFile(string path)
    {
        using var stream = File.OpenRead(path);
        long length = stream.Length;
        if (length > MaxFileBytes) throw new InvalidDataException("The model file is larger than the limit.");
        byte[] bytes = new byte[(int)length];
        try { stream.ReadExactly(bytes); }
        catch (EndOfStreamException error) { throw new InvalidDataException("The model file was truncated while reading.", error); }
        if (stream.ReadByte() != -1) throw new InvalidDataException("The model file changed size while reading.");
        return Read(bytes);
    }

    /// <summary>Reads the model.</summary>
    /// <param name="file">The whole file.</param>
    /// <exception cref="InvalidDataException">The file is truncated, malformed or over a limit.</exception>
    internal static SentencePieceData Read(ReadOnlySpan<byte> file)
    {
        if (file.Length > MaxFileBytes)
        {
            throw new InvalidDataException("The model file is larger than the limit.");
        }

        var pieces = new List<string>();
        var scores = new List<float>();
        var types = new List<byte>();
        string normalizerName = string.Empty;
        byte[] charsmap = [];
        bool dummy = true, strip = true, escape = true, byteFallback = false;
        int modelType = 1, unkId = 0;

        int pos = 0;
        while (pos < file.Length)
        {
            (int field, int wire) = ReadKey(file, ref pos);
            if (field == 1 && wire == 2)
            {
                ReadOnlySpan<byte> body = ReadBytes(file, ref pos);
                if (pieces.Count >= MaxPieces)
                {
                    throw new InvalidDataException("The model lists more pieces than the limit.");
                }

                ReadPiece(body, out string piece, out float score, out byte type);
                pieces.Add(piece);
                scores.Add(score);
                types.Add(type);
            }
            else if (field == 2 && wire == 2)
            {
                ReadOnlySpan<byte> body = ReadBytes(file, ref pos);
                int p = 0;
                while (p < body.Length)
                {
                    (int f, int w) = ReadKey(body, ref p);
                    if (f == 3 && w == 0)
                    {
                        modelType = (int)Math.Min(ReadVarint(body, ref p), int.MaxValue);
                    }
                    else if (f == 35 && w == 0)
                    {
                        byteFallback = ReadVarint(body, ref p) != 0;
                    }
                    else if (f == 40 && w == 0)
                    {
                        unkId = (int)Math.Min(ReadVarint(body, ref p), int.MaxValue);
                    }
                    else
                    {
                        Skip(body, ref p, w);
                    }
                }
            }
            else if (field == 3 && wire == 2)
            {
                ReadOnlySpan<byte> body = ReadBytes(file, ref pos);
                int p = 0;
                while (p < body.Length)
                {
                    (int f, int w) = ReadKey(body, ref p);
                    if (f == 1 && w == 2)
                    {
                        normalizerName = Decode(ReadBytes(body, ref p));
                    }
                    else if (f == 2 && w == 2)
                    {
                        charsmap = ReadBytes(body, ref p).ToArray();
                    }
                    else if (f == 3 && w == 0)
                    {
                        dummy = ReadVarint(body, ref p) != 0;
                    }
                    else if (f == 4 && w == 0)
                    {
                        strip = ReadVarint(body, ref p) != 0;
                    }
                    else if (f == 5 && w == 0)
                    {
                        escape = ReadVarint(body, ref p) != 0;
                    }
                    else
                    {
                        Skip(body, ref p, w);
                    }
                }
            }
            else
            {
                Skip(file, ref pos, wire);
            }
        }

        if (pieces.Count == 0)
        {
            throw new InvalidDataException("The model lists no pieces.");
        }

        if (unkId < 0 || unkId >= pieces.Count)
        {
            throw new InvalidDataException("The model's unknown-piece ID is outside its pieces.");
        }

        return new SentencePieceData
        {
            Pieces = [.. pieces],
            Scores = [.. scores],
            Types = [.. types],
            ModelType = modelType,
            UnkId = unkId,
            ByteFallback = byteFallback,
            NormalizerName = normalizerName,
            Charsmap = charsmap,
            AddDummyPrefix = dummy,
            RemoveExtraWhitespaces = strip,
            EscapeWhitespaces = escape,
        };
    }

    private static void ReadPiece(ReadOnlySpan<byte> body, out string piece, out float score, out byte type)
    {
        piece = string.Empty;
        score = 0;
        type = SentencePieceData.Normal;
        int p = 0;
        while (p < body.Length)
        {
            (int f, int w) = ReadKey(body, ref p);
            if (f == 1 && w == 2)
            {
                ReadOnlySpan<byte> text = ReadBytes(body, ref p);
                if (text.Length > MaxPieceBytes)
                {
                    throw new InvalidDataException("A piece is longer than the limit.");
                }

                piece = Decode(text);
            }
            else if (f == 2 && w == 5)
            {
                if (body.Length - p < 4)
                {
                    throw new InvalidDataException("A score is truncated.");
                }

                score = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(body[p..]));
                p += 4;
            }
            else if (f == 3 && w == 0)
            {
                long t = ReadVarint(body, ref p);
                type = t is >= 1 and <= 6 ? (byte)t : throw new InvalidDataException("A piece has an unknown type.");
            }
            else
            {
                Skip(body, ref p, w);
            }
        }

        if (piece.Length == 0)
        {
            throw new InvalidDataException("A piece is empty.");
        }
    }

    private static string Decode(ReadOnlySpan<byte> bytes)
    {
        try
        {
            return StrictUtf8.GetString(bytes);
        }
        catch (DecoderFallbackException ex)
        {
            throw new InvalidDataException("The model has text that is not valid UTF-8.", ex);
        }
    }

    private static (int Field, int Wire) ReadKey(ReadOnlySpan<byte> data, ref int pos)
    {
        long key = ReadVarint(data, ref pos);
        if (key < 8 || key > int.MaxValue)
        {
            throw new InvalidDataException("The model has an invalid field key.");
        }

        return ((int)(key >> 3), (int)(key & 7));
    }

    private static long ReadVarint(ReadOnlySpan<byte> data, ref int pos)
    {
        long result = 0;
        for (int shift = 0; shift < 70; shift += 7)
        {
            if (pos >= data.Length)
            {
                throw new InvalidDataException("The model is truncated.");
            }

            byte b = data[pos++];
            if (shift < 63)
            {
                result |= (long)(b & 0x7F) << shift;
            }

            if ((b & 0x80) == 0)
            {
                return result < 0 ? long.MaxValue : result;
            }
        }

        throw new InvalidDataException("The model has an over-long integer.");
    }

    private static ReadOnlySpan<byte> ReadBytes(ReadOnlySpan<byte> data, ref int pos)
    {
        long length = ReadVarint(data, ref pos);
        if (length < 0 || length > data.Length - pos)
        {
            throw new InvalidDataException("A length runs past the end of the model.");
        }

        ReadOnlySpan<byte> r = data.Slice(pos, (int)length);
        pos += (int)length;
        return r;
    }

    private static void Skip(ReadOnlySpan<byte> data, ref int pos, int wire)
    {
        switch (wire)
        {
            case 0:
                _ = ReadVarint(data, ref pos);
                break;
            case 1:
                Advance(data, ref pos, 8);
                break;
            case 2:
                _ = ReadBytes(data, ref pos);
                break;
            case 5:
                Advance(data, ref pos, 4);
                break;
            default:
                throw new InvalidDataException("The model uses an unsupported field encoding.");
        }
    }

    private static void Advance(ReadOnlySpan<byte> data, ref int pos, int count)
    {
        if (data.Length - pos < count)
        {
            throw new InvalidDataException("The model is truncated.");
        }

        pos += count;
    }
}
