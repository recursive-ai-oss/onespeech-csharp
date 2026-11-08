namespace OneSpeech.Whisper;

internal static class WhisperCheckpoint
{
    internal static string[] Read(string directory)
    {
        string path = Path.Combine(directory, "upstream.txt");
        if (new FileInfo(path).Length > 1024) throw new ModelIntegrityException("Whisper provenance exceeds 1 KiB.");
        return File.ReadAllLines(path);
    }

    internal static ModelInfo Provenance(string[] lines)
    {
        if (lines.Length != 3) throw new ModelIntegrityException("Expected verified converter provenance in upstream.txt.");
        string upstreamId = lines[0] switch
        {
            "openai/whisper-small-int8" => "openai/whisper-small",
            "openai/whisper-large-v3-turbo-int8" => "openai/whisper-large-v3-turbo",
            _ => lines[0]
        };
        var pin = upstreamId switch
        {
            "openai/whisper-tiny" => ("169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395", "Apache-2.0"),
            "openai/whisper-base" => ("e37978b90ca9030d5170a5c07aadb050351a65bb", "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b", "Apache-2.0"),
            "openai/whisper-large-v3" => ("06f233fe06e710322aca913c1bc4249a0d71fce1", "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95", "Apache-2.0"),
            "openai/whisper-small" => ("973afd24965f72e36ca33b3055d56a652f456b4d", "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b", "Apache-2.0"),
            "openai/whisper-large-v3-turbo" => ("41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1", "MIT"),
            _ => throw new ModelIntegrityException("This converted checkpoint has no verified Whisper licence/provenance entry.")
        };
        if (lines[1] != pin.Item1 || lines[2] != pin.Item2) throw new ModelIntegrityException("Whisper revision or weight digest differs from the supported checkpoint pin; run the pinned converter.");
        return new ModelInfo { Id = lines[0], Source = "https://huggingface.co/" + upstreamId, Revision = pin.Item1, License = pin.Item3, Attribution = "OpenAI Whisper; Copyright (c) 2022 OpenAI", Tier = ModelTier.Tier1, Files = [new ModelFileInfo { Name = "model.safetensors", Sha256 = pin.Item2 }] };
    }
}
