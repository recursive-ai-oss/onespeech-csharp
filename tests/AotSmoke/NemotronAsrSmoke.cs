using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.FastConformer;
using OneSpeech.Vad;

internal static class NemotronAsrSmoke
{
    internal static async Task Run()
    {
        string? cache = Environment.GetEnvironmentVariable("ONESPEECH_NEMOTRON_CACHE"); if (cache is null) return;
        string source = Environment.GetEnvironmentVariable("ONESPEECH_NEMOTRON_SOURCE") ?? throw new InvalidOperationException("Set ONESPEECH_NEMOTRON_SOURCE for the publisher-only model smoke.");
        string path = Environment.GetEnvironmentVariable("ONESPEECH_SMOKE_CLIP") ?? throw new InvalidOperationException("Set ONESPEECH_SMOKE_CLIP.");
        var store = new ModelStore(cache, new DirectoryModelSource(source));
        var resolved = await store.PullAsync("nvidia/nemotron-3.5-asr-streaming-0.6b");
        using (var model = NemotronAsrRecognizer.Load(resolved, new() { Threads = 1 }))
        using (var vad = new SileroVad())
        using (var live = model.CreateLiveSession(vad, "en"))
        {
            var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var document = model.Transcribe(new AudioBuffer(clip.Samples, 16000), "en");
            if (document.Text.Length == 0 || document.Words.IsEmpty || !model.Capabilities.Streaming || document.Models.Single().License != "OpenMDW-1.1" || !document.Models.Single().Attribution.Contains("IN NO EVENT SHALL", StringComparison.Ordinal)) throw new InvalidOperationException("Nemotron model/provenance smoke failed.");
            int partials = 0, finals = 0;
            for (int offset = 0; offset < clip.Samples.Length; offset += 1600)
                foreach (var update in live.Push(new AudioBuffer(clip.Samples.Slice(offset, Math.Min(1600, clip.Samples.Length - offset)), 16000)))
                { if (update is PartialEvent) partials++; if (update is FinalEvent) finals++; }
            foreach (var update in live.Complete()) if (update is FinalEvent) finals++;
            if (partials == 0 || finals == 0) throw new InvalidOperationException("Nemotron genuine partials/finals smoke failed.");
            if (Transcript.FromJson(document.ToJson()).Models.Single().Attribution != resolved.Info.Attribution) throw new InvalidOperationException("Nemotron licence provenance round-trip failed.");
        }
        ModelStore.Default = store;
        int segments = 0;
        await foreach (var segment in Speech.TranscribeStream(path, language: "en", model: "nvidia/nemotron-3.5-asr-streaming-0.6b")) if (segment.Text.Length > 0) segments++;
        if (segments == 0) throw new InvalidOperationException("Nemotron named file streaming registration failed.");
        Console.WriteLine("Nemotron publisher-only cache conversion, native decode, genuine partials and named file streaming NativeAOT smoke passed.");
    }
}
