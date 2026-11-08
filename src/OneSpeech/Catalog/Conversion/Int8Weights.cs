using System.Buffers.Binary;

namespace OneSpeech.Catalog.Conversion;

// Symmetric per-output-channel weights; each family graph supplies its runtime activation ranges.
// No activation statistics or evaluation audio are used during conversion.
internal static class Int8Weights
{
    internal static (byte[] Values, byte[] Scales) Transpose(float[] weights, int outputs, int inputs, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (outputs <= 0 || inputs <= 0 || weights.Length != checked(outputs * inputs))
            throw new ModelIntegrityException("Expected a nonempty rank-two linear weight.");
        byte[] values = new byte[weights.Length];
        byte[] scales = new byte[checked(outputs * 4)];
        for (int o = 0; o < outputs; o++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float maximum = 0;
            for (int i = 0; i < inputs; i++)
            {
                float value = weights[o * inputs + i];
                if (!float.IsFinite(value)) throw new ModelIntegrityException("Cannot quantize a nonfinite weight.");
                maximum = Math.Max(maximum, Math.Abs(value));
            }
            float scale = maximum == 0 ? 1 : maximum / 127;
            if (scale == 0) scale = float.Epsilon;
            BinaryPrimitives.WriteSingleLittleEndian(scales.AsSpan(o * 4), scale);
            for (int i = 0; i < inputs; i++)
                values[i * outputs + o] = unchecked((byte)(sbyte)Math.Clamp(MathF.Round(weights[o * inputs + i] / scale, MidpointRounding.ToEven), -127, 127));
        }
        return (values, scales);
    }
}
