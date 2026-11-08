using OneSpeech;
using OneSpeech.Catalog;
using OneSpeech.FastConformer;

internal static class ParakeetSmoke
{
    internal static void Run()
    {
        string? cache = Environment.GetEnvironmentVariable("ONESPEECH_PARAKEET_CACHE");
        if (cache is null) return;
        string clipPath = Environment.GetEnvironmentVariable("ONESPEECH_SMOKE_CLIP") ?? throw new InvalidOperationException("Set ONESPEECH_SMOKE_CLIP for Parakeet smoke.");
        string? publisherRoot = Environment.GetEnvironmentVariable("ONESPEECH_PARAKEET_SOURCE");
        var store = new ModelStore(cache, publisherRoot is null ? null : new DirectoryModelSource(publisherRoot));
        var resolved = store.PullAsync("nvidia/parakeet-tdt-0.6b-v3").AsTask().GetAwaiter().GetResult();
        using (var recognizer = ParakeetRecognizer.Load(resolved, threads: 4))
        {
            var clip = OneAudio.Audio.Load(clipPath, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            var transcript = recognizer.Transcribe(new AudioBuffer(clip.Samples, 16000), "en");
            if (transcript.Text != "Stuff it into you, his belly counseled him." || transcript.Words.Any(word => word.TimingMethod != TimingMethod.Native || word.Start is null || word.End is null || word.Confidence is null) || transcript.Models.Single().Tier != ModelTier.Tier2)
                throw new InvalidOperationException("Parakeet native recognition failed.");
            bool refused = false;
            try { _ = recognizer.TranscribeStream(new AudioBuffer(new float[320], 16000), "xx").ToArray(); }
            catch (CapabilityNotSupportedException) { refused = true; }
            if (!refused) throw new InvalidOperationException("Parakeet must refuse unsupported languages even on silence.");
            Console.WriteLine("Parakeet fp32 native timing NativeAOT smoke passed.");
        }
        // Use the normal downloader for the tier-1 diarization dependencies. Parakeet is now fully cached.
        ModelStore.Default = new ModelStore(cache);
        var document = Speech.Transcribe(clipPath, language: "en", model: "nvidia/parakeet-tdt-0.6b-v3");
        if (document.Words.IsEmpty || document.Turns.IsEmpty || document.Words.Any(word => word.TimingMethod != TimingMethod.Native) || !document.Models.Any(model => model.Tier == ModelTier.Tier2 && model.License == "CC-BY-4.0" && model.Attribution.Contains("NVIDIA", StringComparison.Ordinal)))
            throw new InvalidOperationException("Parakeet Speech.Transcribe document is incomplete.");
        Console.WriteLine("Parakeet Speech.Transcribe NativeAOT smoke passed.");
    }
}
