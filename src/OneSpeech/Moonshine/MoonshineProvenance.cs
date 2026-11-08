namespace OneSpeech.Moonshine;

internal static class MoonshineProvenance
{
    internal static Catalog.CatalogEntry Verify(string directory, bool converted, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var provenance = File.OpenRead(Path.Combine(directory, "upstream.txt"));
        long length = provenance.Length;
        if (length > 1024) throw new ModelIntegrityException("Moonshine provenance exceeds 1 KiB.");
        var bytes = new byte[(int)length]; provenance.ReadExactly(bytes);
        using var reader = new StringReader(new System.Text.UTF8Encoding(false, true).GetString(bytes));
        string? id = reader.ReadLine(), revision = reader.ReadLine(), sha = reader.ReadLine();
        if (id is null || reader.ReadLine() is not null) throw new ModelIntegrityException("Expected three-line Moonshine provenance.");
        var pin = Get(id);
        if (pin.Revision != revision || pin.Sha256 != sha) throw new ModelIntegrityException("Moonshine provenance differs from the supported pin.");
        var entry = Catalog.ModelCatalog.Default.Get(id);
        var files = converted ? entry.Files.Where(f => f.Name is "config.json" or "tokenizer.json").Concat(entry.ConvertedFiles) : entry.Files;
        var buffer = new byte[65536];
        foreach (var file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var input = File.OpenRead(Path.Combine(directory, file.Name));
            if (input.Length != file.Size) throw new ModelIntegrityException("Moonshine file size differs from the catalog: " + file.Name);
            using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
            int count;
            while ((count = input.Read(buffer)) != 0) { cancellationToken.ThrowIfCancellationRequested(); hash.AppendData(buffer, 0, count); }
            if (Convert.ToHexStringLower(hash.GetHashAndReset()) != file.Sha256) throw new ModelIntegrityException("Moonshine SHA-256 differs from the catalog: " + file.Name + ". Pull or convert the pinned checkpoint again.");
        }
        return entry;
    }

    internal static (string Revision, string Sha256) Get(string id) => (id == "moonshine-ai/moonshine-streaming-tiny-int8" ? "moonshine-ai/moonshine-streaming-tiny" : id) switch
    {
        "moonshine-ai/moonshine-tiny" => ("390624ed33d594443aa4aa221f5b9f283b545b5a", "867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736"),
        "moonshine-ai/moonshine-base" => ("7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf", "e020c79d0a979a7ec099f718ff1cd2f19e92aead230d69654bca5975a8e1b862"),
        "moonshine-ai/moonshine-streaming-tiny" => ("f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a", "0ab3443dabd95a4afb35f0a26155428c03b7c3c450fa6923ffdac4510ba3d040"),
        "moonshine-ai/moonshine-streaming-small" => ("2c036506f23a09c18df5a50057599ba6d9280999", "003635df57b9a47771fd94a6290ab517d7487d0026c165ca9f2ac90411802646"),
        "moonshine-ai/moonshine-streaming-medium" => ("57b843633a8c183cadf6699ffa761377a933a866", "1b431906c919f373c671ffe8516a0125a4e3ffca4a69255c7fd05782abf6a103"),
        _ => throw new CapabilityNotSupportedException("Only the five verified MIT English Moonshine checkpoints are supported. Community-licence checkpoints are not accepted.")
    };
    internal const string Attribution = """
MIT License

Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""";
}
