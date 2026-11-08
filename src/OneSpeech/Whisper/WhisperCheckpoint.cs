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
        var pin = lines[0] switch
        {
            "openai/whisper-tiny" => ("169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395", "Apache-2.0"),
            "openai/whisper-small" => ("973afd24965f72e36ca33b3055d56a652f456b4d", "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b", "Apache-2.0"),
            "openai/whisper-large-v3-turbo" => ("41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1", "MIT"),
            _ => throw new ModelIntegrityException("This converted checkpoint has no verified Whisper licence/provenance entry.")
        };
        if (lines[1] != pin.Item1 || lines[2] != pin.Item2) throw new ModelIntegrityException("Whisper revision or weight digest differs from the supported checkpoint pin; run the pinned converter.");
        return new ModelInfo { Id = lines[0], Source = "https://huggingface.co/" + lines[0], Revision = pin.Item1, License = pin.Item3, Attribution = "OpenAI Whisper; Copyright (c) 2022 OpenAI", Tier = ModelTier.Tier1, Files = [new ModelFileInfo { Name = "model.safetensors", Sha256 = pin.Item2 }] };
    }
}
