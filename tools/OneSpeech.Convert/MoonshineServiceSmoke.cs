using OneSpeech;
using OneSpeech.Catalog;

internal static class MoonshineServiceSmoke
{
    internal static async Task Run(string publisherDirectory, string freshCache, string audioPath)
    {
        if (Directory.Exists(freshCache) && Directory.EnumerateFileSystemEntries(freshCache).Any())
            throw new InvalidOperationException("The first-load smoke requires an empty cache.");
        var previous = ModelStore.Default;
        try
        {
            var source = new PublisherCache(publisherDirectory);
            ModelStore.Default = new ModelStore(freshCache, source);
            int segments = 0;
            await foreach (var segment in Speech.TranscribeStream(audioPath, language: "en", model: "moonshine-ai/moonshine-streaming-tiny"))
                if (segment.Text.Length > 0) segments++;
            if (segments == 0 || source.Opens == 0) throw new InvalidOperationException("First load produced no segment or did not read publisher files.");
            int opens = source.Opens, finals = 0, partials = 0;
            var clip = OneAudio.Audio.Load(audioPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            await foreach (var update in Speech.Listen(Blocks(clip.Samples), language: "en", model: "moonshine-ai/moonshine-streaming-tiny"))
            {
                if (update is FinalEvent) finals++;
                if (update is PartialEvent) partials++;
            }
            if (finals == 0 || partials == 0 || source.Opens != opens) throw new InvalidOperationException("Cached Listen failed or re-read the publisher source.");
            Console.WriteLine($"First-load verified conversion: {opens} publisher files; file segments={segments}; cached live partials={partials}, finals={finals}.");
        }
        finally { ModelStore.Default = previous; }
    }

    private static async IAsyncEnumerable<AudioBuffer> Blocks(ReadOnlyMemory<float> audio)
    {
        for (int offset = 0; offset < audio.Length; offset += 1600)
        {
            await Task.Yield();
            yield return new AudioBuffer(audio.Slice(offset, Math.Min(1600, audio.Length - offset)), 16000);
        }
    }

    private sealed class PublisherCache(string directory) : ModelSource
    {
        internal int Opens { get; private set; }
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (model.Id != "moonshine-ai/moonshine-streaming-tiny") throw new InvalidOperationException("Unexpected source model.");
            Opens++;
            return ValueTask.FromResult<Stream>(File.OpenRead(Path.Combine(directory, file.Name)));
        }
    }
}
