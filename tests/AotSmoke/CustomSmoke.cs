using OneSpeech;
using OneSpeech.Custom;
using OneSpeech.Diarization;

internal static class CustomSmoke
{
    internal static void Run()
    {
        try { _ = PyannoteSegmentation.Load(token: ""); throw new InvalidOperationException("Missing token was accepted."); }
        catch (ModelTokenRequiredException) { }
        try { _ = PyannoteSegmentation.Load(token: "synthetic-smoke-credential"); throw new InvalidOperationException("Unsupported network was accepted."); }
        catch (FeatureUnavailableException) { }
        string? path = Environment.GetEnvironmentVariable("ONESPEECH_CUSTOM_GRAPH");
        if (path is null) return;
        var info = new ModelInfo { Id = "synthetic", Revision = "1", Source = "synthetic", Tier = ModelTier.Tier1, License = "MIT", Attribution = "Synthetic numerical fixture" };
        using var recognizer = new GraphSpeechRecognizer([new() { Path = path, Channels = 3, Stride = 100 }], GraphFrontEnd.RawWaveform, new(["a", " b", ""]), 2, TimeSpan.FromTicks(62500), info);
        var transcript = recognizer.Transcribe(new(new float[400], 16000));
        if (transcript.Text != "a b" || transcript.Words.Length != 2 || transcript.Words[1].End != TimeSpan.FromTicks(187500))
            throw new InvalidOperationException("Custom CTC graph smoke failed.");
        Console.WriteLine("Custom graph recognition and gated segmentation refusal NativeAOT smoke passed.");
    }
}
