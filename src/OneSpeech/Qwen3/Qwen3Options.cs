namespace OneSpeech.Qwen3;

/// <summary>Offline Qwen3-ASR generation settings. Audio is processed in at most 30-second VAD chunks.</summary>
public sealed record Qwen3Options
{
    /// <summary>System-context text for recognition biasing; empty by default. Maximum 4096 tokenizer IDs; special tokens are refused.</summary>
    public string Context { get; init; } = "";
    /// <summary>Maximum generated tokens per chunk, 1–4096; defaults to 512. Reaching the limit throws rather than silently truncating.</summary>
    public int MaxNewTokens { get; init; } = 512;
    /// <summary>OneModel threads per operator; zero selects the runtime default. Range 0–256.</summary>
    public int Threads { get; init; }
    /// <summary>Activation storage precision; Binary32 by default. BFloat16 requires the separately converted bfloat16 graphs.</summary>
    public Qwen3Precision Precision { get; init; }
    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Context);
        if (MaxNewTokens is < 1 or > 4096) throw new ArgumentOutOfRangeException(nameof(MaxNewTokens));
        if (Precision is not (Qwen3Precision.Binary32 or Qwen3Precision.BFloat16)) throw new ArgumentOutOfRangeException(nameof(Precision));
        if (Threads is < 0 or > 256) throw new ArgumentOutOfRangeException(nameof(Threads));
    }
}
