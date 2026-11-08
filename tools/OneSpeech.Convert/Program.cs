using System.Security.Cryptography;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Formats;
using OneSpeech.Whisper;
if (args is ["qwen-precision-repro", var precisionRepro]) { QwenPrecisionProbe.Run(precisionRepro); return 0; }
if (args is ["qwen-cache-paths", var cachePathsRoot]) { QwenTools.CachePaths(cachePathsRoot); return 0; }
if (args is ["qwen-e2e", var e2eCache, var e2eAudio, var e2eJson]) { QwenTools.EndToEnd(e2eCache, e2eAudio, e2eJson); return 0; }
if (args is ["qwen-audio", var canonicalInput, var canonicalOutput]) { QwenTools.CanonicalAudio(canonicalInput, canonicalOutput); return 0; }
if (args is ["qwen-long", var longModel, var longAudio, var longJson, var longPrecision]) { QwenTools.LongForm(longModel, longAudio, longJson, longPrecision == "bf16"); return 0; }
if (args is ["qwen-rebuild", var rebuildModel, var rebuildPrecision])
{
    QwenTools.Rebuild(rebuildModel, rebuildPrecision == "bf16");
    return 0;
}
if (args is ["qwen-reproduce", var reproductionModel, var reproductionPrecision])
{
    QwenTools.Reproduce(reproductionModel, reproductionPrecision == "bf16");
    return 0;
}
if (args is ["qwen-batch-bf16", var bfBatchModel, var bfBatchRoot, var bfBatchSplit, var bfBatchOutput, var bfBatchBegin, var bfBatchCount])
{
    QwenTools.Batch(bfBatchModel, bfBatchRoot, bfBatchSplit, bfBatchOutput, int.Parse(bfBatchBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(bfBatchCount, System.Globalization.CultureInfo.InvariantCulture), OneSpeech.Qwen3.Qwen3Precision.BFloat16);
    return 0;
}
if (args is ["qwen-probe-decode-bf16", var bfProbeModel, var bfProbeData])
{
    QwenTools.ProbeDecoder(bfProbeModel, bfProbeData, true);
    return 0;
}
if (args is ["qwen-convert-bf16", var bf16Input, var bf16Output])
{
    OneSpeech.Catalog.Conversion.QwenConverter.Convert(bf16Input, bf16Output, CancellationToken.None, OneSpeech.Qwen3.Qwen3Precision.BFloat16);
    return 0;
}
if (args is ["qwen-convert", var convertQwenInput, var convertQwenOutput])
{
    OneSpeech.Catalog.Conversion.QwenConverter.Convert(convertQwenInput, convertQwenOutput, CancellationToken.None);
    return 0;
}
if (args is ["qwen-catalog", var catalogModel, var catalogJson])
{
    QwenTools.Catalog(catalogModel, catalogJson);
    return 0;
}
if (args is ["qwen-align-batch", var alignmentModel, var alignmentRoot, var alignmentSplit, var alignmentOutput, var alignmentBegin, var alignmentCount])
{
    QwenTools.AlignBatch(alignmentModel, alignmentRoot, alignmentSplit, alignmentOutput, int.Parse(alignmentBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(alignmentCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["qwen-decode-ids", var tokenDirectory, var tokenIds])
{
    QwenTools.DecodeIds(tokenDirectory, tokenIds);
    return 0;
}
if (args is ["qwen-batch", var batchModel, var batchRoot, var batchSplit, var batchOutput, var batchBegin, var batchCount])
{
    QwenTools.Batch(batchModel, batchRoot, batchSplit, batchOutput, int.Parse(batchBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(batchCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["qwen-probe-decode", var decodeModel, var decodeProbe])
{
    QwenTools.ProbeDecoder(decodeModel, decodeProbe);
    return 0;
}
if (args is ["qwen-probe-encoder", var probeModel, var encoderProbe])
{
    QwenTools.ProbeEncoder(probeModel, encoderProbe);
    return 0;
}
if (args is ["qwen-convert-encoder", var qwenInput])
{
    QwenTools.ConvertEncoder(qwenInput);
    return 0;
}
if (args is ["qwen-positions", var positionProbe])
{
    QwenTools.Positions(positionProbe);
    return 0;
}
if (args is ["qwen-fetch", var qwenName, var qwenDirectory])
{
    await QwenTools.FetchAsync(qwenName, qwenDirectory, CancellationToken.None).ConfigureAwait(false);
    return 0;
}
if (args is ["speaker-extract-ecapa", var ecapaGraph, var ecapaData, var ecapaIds, var ecapaDestination, var ecapaStart, var ecapaCount])
{
    SpeakerBatch.Extract(ecapaGraph, ecapaData, ecapaIds, ecapaDestination, int.Parse(ecapaStart, System.Globalization.CultureInfo.InvariantCulture), int.Parse(ecapaCount, System.Globalization.CultureInfo.InvariantCulture), 192, ecapa: true);
    return 0;
}
if (args is ["speaker-probe-ecapa", var ecapaProbeGraph, var ecapaAudio, var ecapaProbeOutput])
{
    SpeakerProbe.RunEcapa(ecapaProbeGraph, ecapaAudio, ecapaProbeOutput);
    return 0;
}
if (args is ["speaker-convert-ecapa", var ecapaInput, var ecapaOutput])
{
    using var input = File.OpenRead(ecapaInput);
    new EcapaConverter(TensorCheckpoint.Read(input)).Save(ecapaOutput);
    return 0;
}
if (args is ["speaker-ids", var idsTrials, var idsClean, var idsOutput, var idsParity])
{
    SpeakerBatch.PrepareIds(idsTrials, idsClean, idsOutput, idsParity);
    return 0;
}
if (args is ["speaker-extract-camp", var extractGraph, var extractData, var extractIds, var extractOutput, var extractStart, var extractCount, var extractDimension])
{
    SpeakerBatch.Extract(extractGraph, extractData, extractIds, extractOutput, int.Parse(extractStart, System.Globalization.CultureInfo.InvariantCulture), int.Parse(extractCount, System.Globalization.CultureInfo.InvariantCulture), int.Parse(extractDimension, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["speaker-run-features", var featureGraph, var featureInput, var vectorOutput])
{
    SpeakerProbe.RunFeatures(featureGraph, featureInput, vectorOutput);
    return 0;
}
if (args is ["speaker-trials", var cleanRoot, var otherRoot, var trialsOutput, var seed])
{
    SpeakerVerification.Generate(cleanRoot, otherRoot, trialsOutput, int.Parse(seed, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["speaker-score", var trialFile, var oursDirectory, var referenceDirectory, var scoreOutput])
{
    SpeakerVerification.ScoreFiles(trialFile, oursDirectory, referenceDirectory, scoreOutput);
    return 0;
}
if (args is ["speaker-probe", var graphPath, var audioPath, var probeOutput])
{
    SpeakerProbe.Run(graphPath, audioPath, probeOutput);
    return 0;
}
if (args is ["speaker-convert-camp", var campCheckpoint, var campOutput])
{
    using var campInput = File.OpenRead(campCheckpoint);
    new CampPlusConverter(TensorCheckpoint.Read(campInput)).Save(campOutput);
    return 0;
}
if (args is ["speaker-fetch", var root])
{
    await SpeakerTools.FetchAsync(root, CancellationToken.None).ConfigureAwait(false);
    return 0;
}
if (args is ["speaker-inspect", var checkpoint])
{
    using var input = File.OpenRead(checkpoint);
    foreach (var pair in TensorCheckpoint.Read(input))
        Console.WriteLine($"{pair.Key} [{string.Join(',', pair.Value.Shape)}]");
    return 0;
}

if (args.Length < 2 || args[0] is not ("fetch" or "whisper" or "whisper-debug" or "norm-probe" or "block-probe"))
{
    Console.Error.WriteLine("Usage: OneSpeech.Convert fetch <tiny|base|small|large-v3|large-v3-turbo> <directory> | whisper <directory> | speaker-<command> ...");
    return 2;
}
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
if (args[0] == "fetch")
{
    if (args.Length != 3) return 2;
    var pin = args[1] switch
    {
        "tiny" => ("169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"),
        "base" => ("e37978b90ca9030d5170a5c07aadb050351a65bb", "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b"),
        "large-v3" => ("06f233fe06e710322aca913c1bc4249a0d71fce1", "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"),
        "small" => ("973afd24965f72e36ca33b3055d56a652f456b4d", "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"),
        "large-v3-turbo" => ("41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"),
        _ => throw new ArgumentException("Only explicitly pinned checkpoints are accepted.")
    };
    Directory.CreateDirectory(args[2]);
    using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
    foreach (string name in new[] { "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json", "tokenizer_config.json", "special_tokens_map.json", "normalizer.json", "model.safetensors" })
    {
        string path = Path.Combine(args[2], name);
        if (!File.Exists(path))
        {
            using var response = await http.GetAsync($"https://huggingface.co/openai/whisper-{args[1]}/resolve/{pin.Item1}/{name}", HttpCompletionOption.ResponseHeadersRead, cancellation.Token);
            response.EnsureSuccessStatusCode();
            await using (var output = File.Create(path + ".part")) await response.Content.CopyToAsync(output, cancellation.Token);
            File.Move(path + ".part", path);
        }
        await using var input = File.OpenRead(path);
        string sha = Convert.ToHexStringLower(await SHA256.HashDataAsync(input, cancellation.Token));
        if (name == "model.safetensors" && sha != pin.Item2) throw new InvalidDataException("Upstream weight digest mismatch.");
        Console.WriteLine($"{name} {input.Length} {sha}");
    }
    File.WriteAllText(Path.Combine(args[2], "upstream.txt"), $"openai/whisper-{args[1]}\n{pin.Item1}\n{pin.Item2}\n");
    return 0;
}
var config = WhisperConfiguration.Load(args[1]);
string weightPath = Path.Combine(args[1], "model.safetensors");
string pinPath = Path.Combine(args[1], "upstream.txt");
if (!File.Exists(pinPath)) throw new InvalidDataException("Run fetch first to establish verified upstream provenance.");
string[] provenance = WhisperCheckpoint.Read(args[1]);
_ = WhisperCheckpoint.Provenance(provenance);
using (var input = File.OpenRead(weightPath))
    if (Convert.ToHexStringLower(SHA256.HashData(input)) != provenance[2]) throw new InvalidDataException("Weight digest mismatch before conversion.");
using var weights = new SafeTensorFile(File.OpenRead(weightPath));
if (args[0] == "block-probe")
{
    int layer = config.EncoderLayers == 12 ? 6 : 20;
    var block = new WhisperGraphWriter(weights);
    block.Encoder(config, layer, layer + 1);
    block.Save(Path.Combine(args[1], "block-probe.onemodel"), cancellation.Token);
    return 0;
}
if (args[0] == "norm-probe")
{
    var norm = new WhisperGraphWriter(weights);
    norm.NormalizationProbe(config);
    norm.Save(Path.Combine(args[1], "norm-probe.onemodel"), cancellation.Token);
    return 0;
}
foreach (string graph in config.EncoderLayers > 24 ? new[] { "encoder-part0", "encoder", "decoder" } : new[] { "encoder", "decoder" })
{
    var builder = new WhisperGraphWriter(weights, args[0] == "whisper-debug");
    if (graph == "encoder-part0") builder.Encoder(config, 0, 16);
    else if (graph == "encoder") builder.Encoder(config, config.EncoderLayers > 24 ? 16 : 0);
    else builder.Decoder(config);
    string path = Path.Combine(args[1], graph + (args[0] == "whisper-debug" ? ".debug.onemodel" : ".onemodel"));
    builder.Save(path, cancellation.Token);
    using var file = File.OpenRead(path);
    Console.WriteLine($"{graph}.onemodel {file.Length} {Convert.ToHexStringLower(SHA256.HashData(file))}");
}
return 0;
