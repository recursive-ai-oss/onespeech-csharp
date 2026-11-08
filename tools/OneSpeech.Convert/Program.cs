using System.Security.Cryptography;
using OneSpeech.Convert;
using OneSpeech.Formats;
using OneSpeech.Whisper;

if (args.Length < 2 || args[0] is not ("fetch" or "whisper" or "whisper-debug" or "norm-probe" or "block-probe"))
{
    Console.Error.WriteLine("Usage: OneSpeech.Convert fetch <tiny|small|large-v3-turbo> <directory> | whisper <directory>");
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
        "small" => ("973afd24965f72e36ca33b3055d56a652f456b4d", "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"),
        "large-v3-turbo" => ("41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"),
        _ => throw new ArgumentException("Only explicitly pinned checkpoints are accepted.")
    };
    Directory.CreateDirectory(args[2]);
    using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
    foreach (string name in new[] { "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors" })
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
