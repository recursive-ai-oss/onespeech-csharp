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
        string? model = Environment.GetEnvironmentVariable("ONESPEECH_QWEN_SMOKE");
        string? clipPath = Environment.GetEnvironmentVariable("ONESPEECH_SMOKE_CLIP");
        if (model != null && clipPath != null)
        {
            var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            using var recognizer = Qwen3Recognizer.Load(model, new Qwen3Options { Threads = 4 });
            var audio = new OneSpeech.AudioBuffer(clip.Samples, 16000);
            var result = recognizer.Transcribe(audio, "en", CancellationToken.None);
            if (result.Text != "Stuff it into you. His belly counseled him.") throw new InvalidOperationException("Qwen3 ASR NativeAOT probe failed.");
            string? alignment = Environment.GetEnvironmentVariable("ONESPEECH_QWEN_ALIGNER");
            if (alignment != null)
            {
                using var aligner = Qwen3ForcedAligner.Load(alignment, 4);
                var timed = aligner.Align(audio, result.Text, "en", CancellationToken.None);
                if (timed.Words.Length != 8 || timed.Words[0].Start != TimeSpan.FromMilliseconds(320) || timed.Words[0].End != TimeSpan.FromMilliseconds(800))
                    throw new InvalidOperationException("Qwen3 alignment NativeAOT probe failed.");
            }
            Console.WriteLine("Qwen3 ASR and forced alignment NativeAOT smoke passed.");
        }
    }
}
