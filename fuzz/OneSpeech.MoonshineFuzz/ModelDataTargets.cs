using System.Globalization;
using System.Text;
using System.Text.Json;
using OneSpeech;
using OneSpeech.FastConformer;
using OneSpeech.Moonshine;
using OneSpeech.Qwen3;
using OneSpeech.Whisper;
using SharpFuzz;

internal static class ModelDataTargets
{
    internal static bool Run(string[] args)
    {
        string? mode = Environment.GetEnvironmentVariable("ONESPEECH_FUZZ_MODEL_DATA");
        if (mode is null) return false;
        bool tokenizer = mode is "whisper-tokenizer" or "qwen-tokenizer";
        string metadataName = mode == "whisper-tokenizer" ? "added_tokens.json" : "tokenizer_config.json";
        var files = new Dictionary<string, byte[]>(StringComparer.Ordinal)
        {
            ["vocab.json"] = ByteVocabulary(), ["merges.txt"] = "#version: 0.2\n"u8.ToArray(),
            [metadataName] = Encoding.UTF8.GetBytes(mode == "whisper-tokenizer" ? "{}" : """
                {"tokenizer_class":"Qwen2Tokenizer","added_tokens_decoder":{}}
                """),
            ["config.json"] = Encoding.UTF8.GetBytes(mode switch
            {
                "whisper-config" or "whisper-generation" => """
                    {"d_model":4,"encoder_layers":1,"decoder_layers":1,"encoder_attention_heads":2,"decoder_attention_heads":2,"num_mel_bins":80,"vocab_size":256,"encoder_ffn_dim":8,"decoder_ffn_dim":8}
                    """,
                "qwen-config" => """
                    {"thinker_config":{"audio_config":{"num_mel_bins":128,"n_window":50,"n_window_infer":800,"activation_function":"gelu","d_model":4,"encoder_attention_heads":2,"encoder_layers":1,"downsample_hidden_size":4},"text_config":{"hidden_act":"silu","rope_theta":1000000,"rms_norm_eps":0.000001,"attention_bias":false,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"num_hidden_layers":1,"vocab_size":256}}}
                    """,
                _ => "{}"
            }),
            ["generation_config.json"] = """
                {"decoder_start_token_id":2,"eos_token_id":1,"task_to_id":{"transcribe":3,"translate":4},"no_timestamps_token_id":128,"lang_to_id":{"<|en|>":5},"suppress_tokens":[-1,0],"alignment_heads":[[0,0]]}
                """u8.ToArray(),
            ["processor_config.json"] = "{}"u8.ToArray(),
            ["upstream.txt"] = []
        };
        if (args is ["--seed", var corpus])
        {
            Directory.CreateDirectory(corpus);
            // This path runs before instrumentation. Runtime setup must not touch instrumented static initializers.
            if (mode == "nemotron-processor") files["processor_config.json"] = PromptDictionary();
            if (tokenizer)
            {
                foreach (var (name, index) in new[] { ("vocab.json", 0), ("merges.txt", 1), (metadataName, 2) })
                    File.WriteAllBytes(Path.Combine(corpus, "input-" + index.ToString(CultureInfo.InvariantCulture)), [(byte)index, .. files[name]]);
            }
            else if (mode == "provenance")
            {
                foreach (string id in new[] { "openai/whisper-tiny", "moonshine-ai/moonshine-streaming-tiny", "Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ForcedAligner-0.6B" })
                {
                    var entry = OneSpeech.Catalog.ModelCatalog.Default.Get(id);
                    var weight = entry.Files.First(file => file.Name.EndsWith(".safetensors", StringComparison.Ordinal));
                    File.WriteAllText(Path.Combine(corpus, id.Replace('/', '-')), id + "\n" + entry.Revision + "\n" + weight.Sha256 + "\n");
                }
            }
            else File.WriteAllBytes(Path.Combine(corpus, "config.json"), files[mode == "nemotron-processor" ? "processor_config.json" : mode == "whisper-generation" ? "generation_config.json" : "config.json"]);
            return true;
        }
        string directory = Directory.CreateTempSubdirectory("onespeech-model-data-").FullName;
        try
        {
            foreach (var file in files) File.WriteAllBytes(Path.Combine(directory, file.Key), file.Value);
            Fuzzer.LibFuzzer.Run(data =>
            {
                if (tokenizer && data.IsEmpty) return;
                string name = tokenizer ? (data[0] % 3) switch { 0 => "vocab.json", 1 => "merges.txt", _ => metadataName }
                    : mode == "provenance" ? "upstream.txt" : mode == "nemotron-processor" ? "processor_config.json" : mode == "whisper-generation" ? "generation_config.json" : "config.json";
                string path = Path.Combine(directory, name);
                File.WriteAllBytes(path, (tokenizer ? data[1..] : data).ToArray());
                try
                {
                    switch (mode)
                    {
                        case "whisper-tokenizer":
                            Accept(() => { var model = new WhisperTokenizer(directory); _ = model.Decode(model.Encode("a 🙂", true)); }); break;
                        case "qwen-tokenizer":
                            Accept(() => { var model = new Qwen3Tokenizer(directory); _ = model.Decode(model.Encode("a 🙂", true)); }); break;
                        case "whisper-config": Accept(() => _ = WhisperConfiguration.Load(directory)); break;
                        case "qwen-config": Accept(() => _ = QwenConfiguration.Load(directory)); break;
                        case "whisper-generation": Accept(() => _ = WhisperGenerationConfiguration.Load(directory, WhisperConfiguration.Load(directory))); break;
                        case "nemotron-processor": Accept(() => _ = NemotronAsrRecognizer.ReadPrompts(path)); break;
                        case "provenance":
                            Accept(() => _ = WhisperCheckpoint.Provenance(WhisperCheckpoint.Read(directory)));
                            Accept(() => _ = MoonshineProvenance.Verify(directory, false, CancellationToken.None));
                            Accept(() => { using var model = Qwen3Recognizer.Load(directory); });
                            Accept(() => { using var model = Qwen3ForcedAligner.Load(directory); });
                            break;
                        default: throw new ArgumentException("Unknown model-data fuzz target.");
                    }
                }
                finally { File.WriteAllBytes(path, files[name]); }
            });
        }
        finally { Directory.Delete(directory, true); }
        return true;
    }

    private static void Accept(Action action)
    {
        try { action(); }
        catch (InvalidDataException) { }
        catch (JsonException) { }
        catch (ModelIntegrityException) { }
        catch (CapabilityNotSupportedException) { }
        catch (IOException) { }
    }

    private static byte[] ByteVocabulary()
    {
        using var memory = new MemoryStream();
        using (var json = new Utf8JsonWriter(memory))
        {
            json.WriteStartObject(); int extra = 256;
            for (int b = 0; b < 256; b++)
            {
                char character = b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? (char)b : (char)extra++;
                json.WriteNumber(character.ToString(), b);
            }
            json.WriteEndObject();
        }
        return memory.ToArray();
    }
    private static byte[] PromptDictionary()
    {
        using var memory = new MemoryStream();
        using (var json = new Utf8JsonWriter(memory))
        {
            json.WriteStartObject(); json.WriteStartObject("prompt_dictionary"); json.WriteNumber("auto", 101);
            foreach (string language in NemotronAsrRecognizer.SupportedLanguages.Where(language => language.Contains('-', StringComparison.Ordinal))) json.WriteNumber(language, 0);
            json.WriteEndObject(); json.WriteEndObject();
        }
        return memory.ToArray();
    }
}
