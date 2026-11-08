using System.Security.Cryptography;

internal static class OmniTools
{
    private const string Revision = "8e35f0cc28fa6099e0c14d56db85ce0423baa691";
    internal static async Task FetchAsync(string directory, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(directory);
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };
        (string Name, long Size, string Hash)[] files =
        [
            ("omniASR-CTC-300M.pt", 1302113892, "4d31c6a7725a03e90cf26080c67361928755a9f43d322dd574145fbe91c76aec"),
            ("omniASR_tokenizer.model", 87562, "b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe")
        ];
        foreach (var file in files)
        {
            string path = Path.Combine(directory, file.Name);
            if (!File.Exists(path))
            {
                string partial = path + ".partial";
                try
                {
                    var uri = new Uri($"https://huggingface.co/facebook/omniASR-CTC-300M/resolve/{Revision}/{file.Name}");
                    using var response = await client.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                    response.EnsureSuccessStatusCode();
                    await using (var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
                    await using (var output = File.Create(partial))
                    {
                        byte[] buffer = new byte[65536];
                        long total = 0;
                        int count;
                        while ((count = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) != 0)
                        {
                            total += count;
                            if (total > file.Size) throw new InvalidDataException("Model download exceeds pinned size.");
                            await output.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
                        }
                    }
                    Verify(partial, file.Size, file.Hash);
                    File.Move(partial, path);
                }
                finally { if (File.Exists(partial)) File.Delete(partial); }
            }
            Verify(path, file.Size, file.Hash);
            Console.WriteLine($"Verified {file.Name}: {file.Size} bytes, SHA-256 {file.Hash}");
        }
    }

    internal static void Verify(string path, long size, string hash)
    {
        using var stream = File.OpenRead(path);
        if (stream.Length != size || Convert.ToHexStringLower(SHA256.HashData(stream)) != hash)
            throw new InvalidDataException("Model size or SHA-256 mismatch.");
    }
}
