using OneSpeech.Formats;
using OneSpeech.Whisper;
using OneSpeech.Catalog.Conversion;

namespace OneSpeech.Catalog;

// Conversion runs only on already verified publisher files, never on executable model code.
internal static class RuntimeConversion
{
    internal static void Convert(CatalogEntry entry, string input, string output, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (entry.Tier == ModelTier.Tier2)
        {
            if (entry.Id != "nvidia/parakeet-tdt-0.6b-v3" || entry.Family != "parakeet-tdt" || entry.Revision != "541d1f99c6b0c3cd0b11a95167540bb8edefd82b")
                throw new CapabilityNotSupportedException("No publisher-only local converter is registered for this tier-2 checkpoint.");
            ParakeetConverter.Convert(input, output, cancellationToken);
            return;
        }
        if (entry.Tier != ModelTier.Tier1) throw new CapabilityNotSupportedException("Unsupported model tier.");
        if (entry.Family == "whisper")
        {
            var configuration = WhisperConfiguration.Load(input);
            using var weights = new SafeTensorFile(File.OpenRead(Path.Combine(input, "model.safetensors")));
            foreach (var file in entry.ConvertedFiles)
            {
                var graph = new WhisperGraphWriter(weights);
                if (file.Name == "encoder-part0.onemodel") graph.Encoder(configuration, 0, 16);
                else if (file.Name == "encoder.onemodel") graph.Encoder(configuration, configuration.EncoderLayers > 24 ? 16 : 0);
                else if (file.Name == "decoder.onemodel") graph.Decoder(configuration);
                else throw new ModelIntegrityException("Unknown Whisper runtime artifact.");
                graph.Save(Path.Combine(output, file.Name), cancellationToken);
            }
        }
        else if (entry.Family is "qwen3-asr" or "qwen3-aligner")
        {
            QwenConverter.Convert(input, output, cancellationToken);
            if (entry.Family == "qwen3-asr") QwenConverter.Convert(input, output, cancellationToken, OneSpeech.Qwen3.Qwen3Precision.BFloat16);
        }
        else if (entry.Family == "moonshine")
        {
            _ = Moonshine.MoonshineProvenance.Get(entry.Id);
            var configuration = Moonshine.MoonshineConfiguration.Load(input);
            using var weights = new SafeTensorFile(File.OpenRead(Path.Combine(input, "model.safetensors")));
            foreach (var file in entry.ConvertedFiles)
            {
                var graph = new MoonshineGraphWriter(weights);
                if (file.Name == "encoder.onemodel") graph.Encoder(configuration);
                else if (file.Name == "decoder.onemodel") graph.Decoder(configuration);
                else throw new ModelIntegrityException("Unknown Moonshine runtime artifact.");
                graph.Save(Path.Combine(output, file.Name), cancellationToken);
            }
        }
        else if (entry.Id == "iic/speech_campplus_sv_en_voxceleb_16k")
        {
            using var checkpoint = File.OpenRead(Path.Combine(input, "campplus_voxceleb.bin"));
            new CampPlusConverter(TensorCheckpoint.Read(checkpoint)).Save(Path.Combine(output, "campplus-en-fp32.onemodel"));
        }
        else if (entry.Id == "iic/speech_campplus_sv_zh-cn_16k-common")
        {
            using var checkpoint = File.OpenRead(Path.Combine(input, "campplus_cn_common.bin"));
            new CampPlusConverter(TensorCheckpoint.Read(checkpoint)).Save(Path.Combine(output, "campplus-zh-fp32.onemodel"));
        }
        else if (entry.Id == "facebook/omniASR-CTC-300M")
        {
            using var checkpoint = File.OpenRead(Path.Combine(input, "omniASR-CTC-300M.pt"));
            new OmniAsrConverter(TensorCheckpoint.Read(checkpoint, largeModel: true)).Save(Path.Combine(output, "ctc-fp32.onemodel"));
        }
        else throw new CapabilityNotSupportedException($"No local converter is registered for '{entry.Id}'.");
        cancellationToken.ThrowIfCancellationRequested();
    }

    internal static void WriteProvenance(CatalogEntry entry, string directory)
    {
        if (entry.Family is not ("whisper" or "moonshine" or "qwen3-asr" or "qwen3-aligner")) return;
        string sha = entry.Files.First(f => f.Name.EndsWith(".safetensors", StringComparison.Ordinal)).Sha256;
        string path = Path.Combine(directory, "upstream.txt");
        string text = $"{entry.Id}\n{entry.Revision}\n{sha}\n";
        if (File.Exists(path) && new FileInfo(path).Length <= 1024 && File.ReadAllText(path) == text) return;
        string temporary = path + ".part-" + Guid.NewGuid().ToString("N");
        try { File.WriteAllText(temporary, text); File.Move(temporary, path, overwrite: true); }
        finally { File.Delete(temporary); }
    }
}
