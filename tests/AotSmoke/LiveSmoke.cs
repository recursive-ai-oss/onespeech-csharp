using OneSpeech;
using OneSpeech.Vad;

internal static class LiveSmoke
{
    internal static async Task Run(SileroVad vad, SpeechRecognizer recognizer, string path, bool expectPartials = false)
    {
        var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
        using var session = recognizer.CreateLiveSession(vad, "en");
        TimeSpan clock = TimeSpan.Zero;
        int finals = 0, partials = 0;
        await foreach (var update in session.Listen(Blocks(clip.Samples), CancellationToken.None))
        {
            if (update.AudioTime < clock) throw new InvalidOperationException("Live audio clock regressed.");
            clock = update.AudioTime;
            if (update is PartialEvent) { partials++; if (!expectPartials) throw new InvalidOperationException("Offline recognizer emitted a partial."); }
            if (update is FinalEvent final)
            {
                if (final.Segment.End > update.AudioTime || final.Segment.Text.Length == 0) throw new InvalidOperationException("Invalid live Final.");
                finals++;
            }
        }
        if (finals == 0 || expectPartials && partials == 0 || session.AudioTime != TimeSpan.FromSeconds(clip.Samples.Length / 16000d))
            throw new InvalidOperationException("Live fixture did not produce a final on the sample clock.");
        Console.WriteLine("VAD-endpointed live fixture NativeAOT passed.");
    }

    private static async IAsyncEnumerable<AudioBuffer> Blocks(ReadOnlyMemory<float> samples)
    {
        for (int offset = 0; offset < samples.Length; offset += 1600)
        {
            await Task.Yield();
            yield return new AudioBuffer(samples.Slice(offset, Math.Min(1600, samples.Length - offset)), 16000);
        }
    }
}
