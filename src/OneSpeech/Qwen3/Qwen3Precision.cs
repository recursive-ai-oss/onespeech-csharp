namespace OneSpeech.Qwen3;

/// <summary>Qwen neural arithmetic storage. OneModel computes floating-point operators in binary32.</summary>
public enum Qwen3Precision
{
    /// <summary>Published weights expanded exactly to binary32; activations remain binary32.</summary>
    Binary32,
    /// <summary>Published bfloat16 weights with bfloat16 rounding at neural-operator boundaries; accumulation remains binary32.</summary>
    BFloat16
}

internal static class QwenRounding
{
    internal static float BFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return BitConverter.UInt32BitsToSingle(0x7fc00000);
        return BitConverter.UInt32BitsToSingle((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
    }
}
