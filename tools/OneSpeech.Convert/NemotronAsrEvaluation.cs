using System.Text.Json;
using System.Collections.Immutable;
using OneSpeech.Decoding;
using OneSpeech.Live;
using OneSpeech;
using OneSpeech.FastConformer;

internal static class NemotronAsrEvaluation
{
    internal static ModelInfo Info()
    {
        var e = OneSpeech.Catalog.ModelCatalog.Default.Get(NemotronAsrRecognizer.Identifier);
        return new() { Id = e.Id, Revision = e.Revision, Source = e.Source, Tier = e.Tier, License = e.License, Attribution = e.Attribution, Files = [.. e.Files.Concat(e.ConvertedFiles).Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha256 })] };
    }
    internal static void Batch(string directory, string corpus, string output, int chunkMilliseconds, int begin, int count, bool publishedEvents = false)
    {
        Directory.CreateDirectory(output);
        var info = Info();
        using var recognizer = NemotronAsrRecognizer.Load(new(directory, info), new() { ChunkMilliseconds = chunkMilliseconds, Threads = 4 });
        foreach (string path in Directory.EnumerateFiles(corpus, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(200).Skip(begin).Take(count))
        {
            string id = Path.GetFileNameWithoutExtension(path), target = Path.Combine(output, id + ".json"); if (File.Exists(target)) continue;
            var clip = OneAudio.Audio.Load(path, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
            using var state = new NemotronAsrState(recognizer, recognizer.PromptId("en"), CancellationToken.None, padFinalChunk: chunkMilliseconds != 320);
            List<double> availability = []; ImmutableArray<FrameToken> result; string text;
            if (publishedEvents)
            {
                var recording = new Recording(state, availability);
                // Corpus files are already utterances; isolate the documented streaming recognizer,
                // not differences between endpoint detectors. Use the real event publication policy.
                using var session = new UtteranceLiveSession(recognizer, (_, _) => 1, recording, "en");
                List<string> finals = [];
                int block = chunkMilliseconds * 16;
                for (int offset = 0; offset < clip.Samples.Length; offset += block)
                {
                    int length = Math.Min(block, clip.Samples.Length - offset); recording.AvailableAt = (offset + length) / 16000d;
                    foreach (var update in session.Push(new AudioBuffer(clip.Samples.Slice(offset, length), 16000))) if (update is FinalEvent final) finals.Add(final.Segment.Text);
                }
                recording.AvailableAt = clip.Samples.Length / 16000d;
                foreach (var update in session.Complete()) if (update is FinalEvent final) finals.Add(final.Segment.Text);
                result = [.. recording.Accepted]; text = string.Join(' ', finals);
            }
            else
            {
                int emitted = 0;
                for (int offset = 0; offset < clip.Samples.Length; offset += 160)
                {
                    int length = Math.Min(160, clip.Samples.Length - offset);
                    state.Push(clip.Samples.Span.Slice(offset, length), false, CancellationToken.None);
                    while (emitted < state.TokenCount) { availability.Add((offset + length) / 16000d); emitted++; }
                }
                state.Push([], true, CancellationToken.None); result = state.Tokens;
                while (emitted < result.Length) { availability.Add(clip.Samples.Length / 16000d); emitted++; }
                text = recognizer.Snapshot(result, TimeSpan.Zero, TimeSpan.FromSeconds(clip.Samples.Length / 16000d), "en", CancellationToken.None).Text;
            }
            using (var stream = File.Create(target + ".partial"))
            using (var json = new Utf8JsonWriter(stream))
            {
                json.WriteStartObject(); json.WriteString("id", id); json.WriteString("text", text); json.WriteNumber("duration", clip.Samples.Length / 16000d);
                json.WriteStartArray("tokens"); foreach (var token in result) json.WriteNumberValue(token.Id); json.WriteEndArray();
                json.WriteStartArray("frames"); foreach (var token in result) json.WriteNumberValue(token.StartFrame); json.WriteEndArray();
                json.WriteStartArray("available"); foreach (double value in availability) json.WriteNumberValue(value); json.WriteEndArray();
                json.WriteEndObject();
            }
            File.Move(target + ".partial", target); Console.WriteLine(id);
        }
    }
    private sealed class Recording(NemotronAsrState state, List<double> available) : IIncrementalUtteranceRecognizer
    {
        internal readonly List<FrameToken> Accepted = [];
        internal double AvailableAt { get; set; }
        private int copied;
        public int PartialCadenceSamples => state.PartialCadenceSamples;
        public Segment Decode(ReadOnlyMemory<float> audio, string? language, bool final, TimeSpan start, TimeSpan end, CancellationToken cancellationToken)
        {
            var segment = state.Decode(audio, language, final, start, end, cancellationToken); var tokens = state.Tokens;
            int offset = (int)Math.Round(start.TotalSeconds / .08);
            while (copied < tokens.Length)
            {
                var token = tokens[copied++]; Accepted.Add(token with { StartFrame = token.StartFrame + offset, EndFrame = token.EndFrame + offset }); available.Add(AvailableAt);
            }
            return segment;
        }
        public void Reset() { state.Reset(); copied = 0; }
        public void Dispose() { } // The batch owns the borrowed state.
    }
}

internal static class NemotronAsrChecks
{
    internal static void Vocabulary(string directory, string fixtures)
    {
        var vocabulary = new ParakeetVocabulary(Path.Combine(directory, "tokenizer.json"), 13087);
        using var document = JsonDocument.Parse(File.ReadAllBytes(fixtures)); int count = 0, errors = 0;
        foreach (var row in document.RootElement.EnumerateArray())
        {
            int[] ids = row.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            string expected = row.GetProperty("text").GetString()!, actual = vocabulary.Decode(ids);
            if (expected != actual) { if (errors < 8) Console.WriteLine($"Mismatch [{string.Join(',', ids)}] expected=[{expected}] actual=[{actual}]"); errors++; } count++;
        }
        Console.WriteLine($"Vocabulary {count} cases, {errors} mismatches"); if (errors != 0) throw new InvalidDataException("Vocabulary mismatch.");
    }
    internal static void Prepare(string corpus, string ours, string reference, string output, int count = 200)
    {
        Directory.CreateDirectory(output);
        var truth = OneSpeech.Evaluation.Hypotheses.LibriSpeech(corpus).ToDictionary(x => x.Id, StringComparer.Ordinal);
        List<OneSpeech.Evaluation.Utterance> expected = [], actual = [], oracle = []; int equal = 0, total = 0;
        foreach (string path in Directory.EnumerateFiles(corpus, "*.flac", SearchOption.AllDirectories).OrderBy(Path.GetFileNameWithoutExtension, StringComparer.Ordinal).Take(count))
        {
            string id = Path.GetFileNameWithoutExtension(path);
            using var a = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(ours, id + ".json")));
            using var b = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, id + ".json")));
            expected.Add(truth[id]); actual.Add(new(id, a.RootElement.GetProperty("text").GetString()!)); oracle.Add(new(id, b.RootElement.GetProperty("text").GetString()!));
            int[] left = a.RootElement.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray(), right = b.RootElement.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            if (left.SequenceEqual(right)) equal++; else Console.WriteLine("Token divergence: " + id);
            total += right.Length;
        }
        OneSpeech.Evaluation.Hypotheses.Write(Path.Combine(output, "truth.jsonl"), expected);
        OneSpeech.Evaluation.Hypotheses.Write(Path.Combine(output, "ours.jsonl"), actual);
        OneSpeech.Evaluation.Hypotheses.Write(Path.Combine(output, "reference.jsonl"), oracle);
        Console.WriteLine($"Token identity: {equal}/{count} utterances, {total} reference tokens");
    }
    internal static void Catalog(string publisher, string converted, string catalogPath, string licensePath, string output)
    {
        var root = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllBytes(catalogPath))!.AsObject();
        string id = NemotronAsrRecognizer.Identifier, revision = NemotronAsrRecognizer.Revision, source = "https://huggingface.co/" + id;
        System.Text.Json.Nodes.JsonObject FileEntry(string path, string url)
        {
            using var file = File.OpenRead(path);
            return new() { ["name"] = Path.GetFileName(path), ["size"] = file.Length, ["sha256"] = System.Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(file)), ["upstreamUrl"] = url, ["mirrorUrl"] = null };
        }
        var files = new System.Text.Json.Nodes.JsonArray();
        foreach (string name in new[] { "model.safetensors", "config.json", "processor_config.json", "tokenizer.json" }) files.Add(FileEntry(Path.Combine(publisher, name), source + "/resolve/" + revision + "/" + name));
        var outputs = new System.Text.Json.Nodes.JsonArray();
        foreach (string name in new[] { "embedding.f32", "encoder-step0.onemodel", "encoder-step1.onemodel", "encoder-tail0.onemodel", "prompt.onemodel", "predictor.onemodel", "joint.onemodel" }) outputs.Add(FileEntry(Path.Combine(converted, name), source + "/resolve/" + revision + "/model.safetensors"));
        var languages = new System.Text.Json.Nodes.JsonArray(); foreach (string code in NemotronAsrRecognizer.SupportedLanguages) languages.Add(code);
        root["models"]!.AsArray().Add(new System.Text.Json.Nodes.JsonObject { ["id"] = id, ["family"] = "nemotron-asr", ["source"] = source, ["revision"] = revision, ["files"] = files, ["convertedFiles"] = outputs, ["tier"] = "Tier2", ["license"] = "OpenMDW-1.1", ["gated"] = false, ["isDefault"] = false, ["attribution"] = "Developed by NVIDIA. NVIDIA Nemotron 3.5 ASR. " + source + ". Licensed under OpenMDW-1.1 (https://openmdw.ai/license/1-1/). Converted locally to OneModel format; weights unchanged. Never redistributed by OneSpeech.\n\n" + File.ReadAllText(licensePath), ["capabilities"] = new System.Text.Json.Nodes.JsonObject { ["languages"] = languages, ["nativeTimestamps"] = true, ["streaming"] = true } });
        using var stream = File.Create(output); using var writer = new Utf8JsonWriter(stream, new() { Indented = true }); root.WriteTo(writer);
    }
}

internal static class NemotronAsrLatency
{
    // Same native acoustic clock for both engines; finalized words are closed by the next word or EOF.
    internal static void Run(string directory, string baseline, string hypothesis, string? timingTruth = null)
    {
        var truth = timingTruth is null ? null : OneSpeech.Evaluation.Hypotheses.Read(timingTruth);
        var vocabulary = new ParakeetVocabulary(Path.Combine(directory, "tokenizer.json"), 13087);
        List<double> delays = [], finals = []; int utterances = 0, unmatched = 0;
        foreach (string path in Directory.EnumerateFiles(baseline, "*.json").Order(StringComparer.Ordinal))
        {
            using var b = JsonDocument.Parse(File.ReadAllBytes(path)); if (!b.RootElement.TryGetProperty("raw_ids", out var raw)) continue;
            using var h = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(hypothesis, Path.GetFileName(path))));
            string id = b.RootElement.GetProperty("id").GetString()!;
            if (truth is not null && !truth.ContainsKey(id)) continue;
            double duration = b.RootElement.GetProperty("duration").GetDouble();
            List<OneSpeech.Decoding.FrameToken> reference = []; int frame = 0;
            foreach (var item in raw.EnumerateArray())
            {
                int tokenId = item.GetInt32(); if (tokenId == 13087) { frame++; continue; }
                if (!vocabulary.IsSpecial(tokenId)) reference.Add(new(tokenId, frame, frame + 1, 1));
            }
            var source = reference.ToArray(); var expected = vocabulary.Words(source);
            double[] ends = expected.Select(word => Math.Min(duration, source[word.FirstToken + word.TokenCount - 1].EndFrame * .08)).ToArray();
            if (truth is not null)
            {
                var words = truth[id].Words?.Where(word => word.End is not null).ToArray() ?? throw new InvalidDataException("Missing timing truth words.");
                expected = words.Select(word => new OneSpeech.Decoding.WordTokenRange(word.Text, 0, 0)).ToArray();
                ends = words.Select(word => word.End!.Value).ToArray();
            }
            int[] ids = h.RootElement.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            double[] available = h.RootElement.GetProperty("available").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            if (ids.Length != available.Length) throw new InvalidDataException("Availability must be recorded for every nonblank token.");
            var target = ids.Select(x => new OneSpeech.Decoding.FrameToken(x, 0, 1, 1)).ToArray(); var actual = vocabulary.Words(target);
            if (expected.Length > 1000 || actual.Length > 1000) throw new InvalidDataException("Latency alignment exceeds word bound.");
            string N(string text) => OneSpeech.Evaluation.Normalizer.Normalize(text, "en");
            var cost = new int[expected.Length + 1, actual.Length + 1];
            for (int i = 0; i <= expected.Length; i++) cost[i, 0] = i;
            for (int j = 0; j <= actual.Length; j++) cost[0, j] = j;
            for (int i = 1; i <= expected.Length; i++) for (int j = 1; j <= actual.Length; j++) cost[i, j] = Math.Min(cost[i - 1, j - 1] + (N(expected[i - 1].Text) == N(actual[j - 1].Text) ? 0 : 1), Math.Min(cost[i - 1, j] + 1, cost[i, j - 1] + 1));
            int r = expected.Length, c = actual.Length;
            while (r > 0 || c > 0)
            {
                bool match = r > 0 && c > 0 && N(expected[r - 1].Text) == N(actual[c - 1].Text);
                if (r > 0 && c > 0 && cost[r, c] == cost[r - 1, c - 1] + (match ? 0 : 1))
                {
                    if (match && N(expected[r - 1].Text).Length > 0)
                    {
                        double end = ends[r - 1];
                        double emitted = c == actual.Length ? duration : available[actual[c].FirstToken];
                        delays.Add((emitted - end) * 1000);
                    }
                    else unmatched++;
                    r--; c--;
                }
                else if (r > 0 && cost[r, c] == cost[r - 1, c] + 1) { unmatched++; r--; }
                else { unmatched++; c--; }
            }
            if (ends.Length > 0) finals.Add((duration - ends[^1]) * 1000);
            utterances++;
        }
        delays.Sort(); finals.Sort();
        double Median(List<double> values) => values.Count % 2 == 0 ? (values[values.Count / 2 - 1] + values[values.Count / 2]) / 2 : values[values.Count / 2];
        string clock = timingTruth is null ? "common offline native ends" : "MFA word ends (available truth subset)";
        Console.WriteLine($"Stable-word availability vs {clock}: n={delays.Count}, utterances={utterances}, unmatched={unmatched}, meanMs={delays.Average():F3}, medianMs={Median(delays):F3}, negative={delays.Count(x => x < 0)}");
        Console.WriteLine($"EOF Final availability vs last {clock}: n={finals.Count}, meanMs={finals.Average():F3}, medianMs={Median(finals):F3}");
    }
}
