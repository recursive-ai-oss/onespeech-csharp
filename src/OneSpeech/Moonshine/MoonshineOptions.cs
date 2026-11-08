namespace OneSpeech.Moonshine;

/// <summary>Greedy fp32 Moonshine recognition settings. Translation, biasing and word timing are not provided.</summary>
public sealed record MoonshineOptions
{
    /// <summary>Inference threads, default one, valid range 1–256.</summary>
    public int Threads { get; init; } = 1;
    /// <summary>Maximum new tokens per audio second, default 6.5 using the published model card rate, excluding BOS, finite in (0,32]. EOS can stop earlier.</summary>
    public double TokensPerSecond { get; init; } = 6.5;
    internal void Validate()
    {
        if (Threads is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(Threads));
        if (!double.IsFinite(TokensPerSecond) || TokensPerSecond <= 0 || TokensPerSecond > 32) throw new ArgumentOutOfRangeException(nameof(TokensPerSecond));
    }
}
