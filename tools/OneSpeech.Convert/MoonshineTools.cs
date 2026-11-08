using System.Security.Cryptography;
using OneSpeech.Formats;

internal static class MoonshineTools
{
    internal static void TokenizerCorpus(string directory, string path)
    {
        var tokenizer = new OneSpeech.Moonshine.MoonshineTokenizer(directory);
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(path));
        int count = 0;
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            string text = item.GetProperty("text").GetString()!;
            int[] ids = tokenizer.Encode(text);
            if (!ids.SequenceEqual(item.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32())) || tokenizer.Decode(ids, false) != item.GetProperty("decoded").GetString() || tokenizer.Decode(ids) != item.GetProperty("withoutSpecials").GetString())
                throw new InvalidDataException("Tokenizer mismatch at case " + count + " text=" + text);
            count++;
        }
        foreach (var item in document.RootElement.GetProperty("decodeCases").EnumerateArray())
        {
            int[] ids = item.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray();
            if (tokenizer.Decode(ids, false) != item.GetProperty("decoded").GetString() || tokenizer.Decode(ids) != item.GetProperty("withoutSpecials").GetString())
                throw new InvalidDataException("Decoder mismatch: " + string.Join(",", ids) + " ours=" + tokenizer.Decode(ids) + " reference=" + item.GetProperty("withoutSpecials").GetString());
            count++;
        }
        Console.WriteLine(count + " tokenizer cases, zero mismatches.");
    }

    internal static void Reproduce(string catalog, string root)
    {
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(catalog));
        foreach (var json in document.RootElement.EnumerateArray())
        {
            var entry = OneSpeech.Catalog.CatalogEntry.FromJson(System.Text.Encoding.UTF8.GetBytes(json.GetRawText()));
            string directory = Path.Combine(root, entry.Id[(entry.Id.LastIndexOf('/') + 1)..]);
            string output = Path.Combine(directory, "reproduced"); Directory.CreateDirectory(output);
            try
            {
                OneSpeech.Catalog.RuntimeConversion.Convert(entry, directory, output, CancellationToken.None);
                foreach (var file in entry.ConvertedFiles)
                {
                    using var input = File.OpenRead(Path.Combine(output, file.Name));
                    string sha = Convert.ToHexStringLower(SHA256.HashData(input));
                    if (input.Length != file.Size || sha != file.Sha256) throw new InvalidDataException("Runtime conversion is not reproducible: " + entry.Id + " " + file.Name);
                    Console.WriteLine(entry.Id + " " + file.Name + " " + input.Length + " " + sha);
                }
            }
            finally { Directory.Delete(output, true); }
        }
    }

    internal static void Catalog(string root, string destination)
    {
        using var stream = File.Create(destination);
        using var json = new System.Text.Json.Utf8JsonWriter(stream, new System.Text.Json.JsonWriterOptions { Indented = true });
        json.WriteStartArray();
        foreach (string name in new[] { "tiny", "base", "streaming-tiny", "streaming-small", "streaming-medium" })
        {
            string id = "moonshine-ai/moonshine-" + name, directory = Path.Combine(root, "moonshine-" + name);
            var pin = OneSpeech.Moonshine.MoonshineProvenance.Get(id);
            json.WriteStartObject(); json.WriteString("id", id); json.WriteString("family", "moonshine"); json.WriteString("source", "https://huggingface.co/" + id); json.WriteString("revision", pin.Revision);
            json.WriteStartArray("files");
            foreach (string file in new[] { "model.safetensors", "config.json", "tokenizer.json", "preprocessor_config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json" })
                if (File.Exists(Path.Combine(directory, file))) Write(file, "https://huggingface.co/" + id + "/resolve/" + pin.Revision + "/" + file);
            json.WriteEndArray(); json.WriteStartArray("convertedFiles");
            foreach (string file in new[] { "encoder.onemodel", "decoder.onemodel" }) Write(file, "https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/models-v1/moonshine-" + name + "-" + file);
            json.WriteEndArray(); json.WriteString("tier", "Tier1"); json.WriteString("license", "MIT"); json.WriteString("attribution", OneSpeech.Moonshine.MoonshineProvenance.Attribution); json.WriteBoolean("gated", false); json.WriteBoolean("isDefault", name == "streaming-tiny");
            json.WriteStartObject("capabilities"); json.WriteStartArray("languages"); json.WriteStringValue("en"); json.WriteEndArray(); json.WriteBoolean("streaming", name.StartsWith("streaming-", StringComparison.Ordinal)); json.WriteEndObject(); json.WriteEndObject();
            void Write(string file, string url)
            {
                using var input = File.OpenRead(Path.Combine(directory, file));
                string sha = Convert.ToHexStringLower(SHA256.HashData(input));
                json.WriteStartObject(); json.WriteString("name", file); json.WriteNumber("size", input.Length); json.WriteString("sha256", sha); json.WriteString("upstreamUrl", url); json.WriteNull("mirrorUrl"); json.WriteEndObject();
            }
        }
        json.WriteEndArray();
    }

    internal static async Task FetchAsync(string name, string directory, CancellationToken cancellationToken)
    {
        var pin = OneSpeech.Moonshine.MoonshineProvenance.Get("moonshine-ai/moonshine-" + name);
        Directory.CreateDirectory(directory);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };
        foreach (string file in new[] { "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors" })
        {
            string path = Path.Combine(directory, file);
            if (!File.Exists(path))
            {
                using var response = await http.GetAsync($"https://huggingface.co/moonshine-ai/moonshine-{name}/resolve/{pin.Item1}/{file}", HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                if (response.StatusCode == System.Net.HttpStatusCode.NotFound && file is not ("model.safetensors" or "config.json" or "tokenizer.json")) continue;
                response.EnsureSuccessStatusCode();
                await using (var output = File.Create(path + ".part")) await response.Content.CopyToAsync(output, cancellationToken).ConfigureAwait(false);
                File.Move(path + ".part", path);
            }
            await using var input = File.OpenRead(path);
            string sha = Convert.ToHexStringLower(await SHA256.HashDataAsync(input, cancellationToken).ConfigureAwait(false));
            if (file == "model.safetensors" && sha != pin.Item2) throw new OneSpeech.ModelIntegrityException("Moonshine upstream weight digest mismatch.");
            Console.WriteLine($"{file} {input.Length} {sha}");
        }
        await File.WriteAllTextAsync(Path.Combine(directory, "upstream.txt"), $"moonshine-ai/moonshine-{name}\n{pin.Item1}\n{pin.Item2}\n", cancellationToken).ConfigureAwait(false);
    }

    internal static void Inspect(string directory)
    {
        _ = OneSpeech.Moonshine.MoonshineProvenance.Verify(directory, converted: false, CancellationToken.None);
        using var input = File.OpenRead(Path.Combine(directory, "model.safetensors"));
        using var weights = new SafeTensorFile(input);
        foreach (var pair in weights.Tensors) Console.WriteLine($"{pair.Key} [{string.Join(',', pair.Value.Shape)}]");
        if (weights.Tensors.ContainsKey("model.encoder.embedder.comp.log_k")) Console.WriteLine("log_k=" + weights.ReadFloat32("model.encoder.embedder.comp.log_k", CancellationToken.None)[0]);
    }
}
