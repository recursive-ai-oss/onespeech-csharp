using OneSpeech.Qwen3;

internal static class Qwen3Smoke
{
    internal static void Run()
    {
        string? root = Environment.GetEnvironmentVariable("ONESPEECH_QWEN_TOKENIZERS");
        if (root is null) return;
        foreach (string family in new[] { "asr", "asr-1.7b", "aligner" })
        {
            var tokenizer = new Qwen3Tokenizer(Path.Combine(root, family));
            int[] ids = tokenizer.Encode("Hello world!", cancellationToken: CancellationToken.None);
            if (!ids.AsSpan().SequenceEqual([9707, 1879, 0]) || tokenizer.Decode(ids) != "Hello world!" ||
                tokenizer.Decode(tokenizer.Encode("e\u0301", cancellationToken: CancellationToken.None)) != "é" ||
                tokenizer.GetToken("<|audio_pad|>") != 151676)
                throw new InvalidOperationException("Qwen3 tokenizer NativeAOT smoke failed.");
        }
        Console.WriteLine("Qwen3 pinned tokenizers NativeAOT smoke passed.");
    }
}
