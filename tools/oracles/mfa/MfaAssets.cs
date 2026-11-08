internal static class MfaAssets
{
    internal static async Task FetchAsync(string directory, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "ATTRIBUTION.txt"), "English MFA acoustic model v3.0.0 and English MFA dictionary v3.0.0 — Michael McAuliffe and Morgan Sonderegger (2024), Montreal Corpus Tools. CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/ . Unmodified release assets from https://github.com/MontrealCorpusTools/mfa-models/releases ; release-tag commit f76a59f7491eadda0fee212b329521e20e349e75. Local evaluation copies; never redistributed by OneSpeech.\n");
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        (string Kind, string Name, long Size, string Hash)[] pins =
        [
            ("acoustic", "english_mfa.zip", 92175348, "a5c26c360b741e9a8e692416d942dae1fb2a8957dccc0826c4d6a87c98945f73"),
            ("dictionary", "english_mfa.dict", 7862165, "30da2ad9168290da124369edb8011ebca5e9a6f0d4b0ad5e0fc33fa0fe12ca77")
        ];
        foreach (var pin in pins)
        {
            string path = Path.Combine(directory, pin.Name);
            if (!File.Exists(path))
            {
                string partial = path + ".partial";
                try
                {
                    using var response = await client.GetAsync(new Uri($"https://github.com/MontrealCorpusTools/mfa-models/releases/download/{pin.Kind}-english_mfa-v3.0.0/{pin.Name}"), HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                    response.EnsureSuccessStatusCode();
                    await using (var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
                    await using (var output = File.Create(partial))
                    {
                        byte[] buffer = new byte[65536]; long total = 0; int count;
                        while ((count = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) != 0)
                        {
                            total += count;
                            if (total > pin.Size) throw new InvalidDataException("MFA asset exceeds pinned size.");
                            await output.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
                        }
                    }
                    OmniTools.Verify(partial, pin.Size, pin.Hash);
                    File.Move(partial, path);
                }
                finally { if (File.Exists(partial)) File.Delete(partial); }
            }
            OmniTools.Verify(path, pin.Size, pin.Hash);
            Console.WriteLine($"{pin.Name} {pin.Size} {pin.Hash}");
        }
    }

}
