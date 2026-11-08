using System.Net.Http;
using System.Security.Cryptography;

internal static class SpeakerTools
{
    internal static readonly (string Id, string File, string Url, string Hash)[] Weights =
    [
        ("ecapa", "embedding_model.ckpt", "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/embedding_model.ckpt", "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"),
        ("camp-en", "campplus_voxceleb.bin", "https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_en_voxceleb_16k/repo?Revision=032b8131a7ad812f87061955ca974c99060c5a03&FilePath=campplus_voxceleb.bin", "5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de"),
        ("camp-zh", "campplus_cn_common.bin", "https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh-cn_16k-common/repo?Revision=013c18884dffb82846cf472dd06dd4a08d97e2cd&FilePath=campplus_cn_common.bin", "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"),
        ("eres2netv2", "pretrained_eres2netv2.ckpt", "https://modelscope.cn/api/v1/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/repo?Revision=1cf80d41fb3435bd3d8df185b5c423333b2db42a&FilePath=pretrained_eres2netv2.ckpt", "0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c")
    ];

    internal static async Task FetchAsync(string root, CancellationToken cancellationToken)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        foreach (var model in Weights)
        {
            string directory = Path.Combine(root, model.Id);
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory, model.File);
            if (!File.Exists(path))
            {
                string partial = path + ".partial";
                try
                {
                    using var response = await client.GetAsync(new Uri(model.Url), HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
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
                            if (total > 128 * 1024 * 1024) throw new InvalidDataException("Checkpoint exceeds download bound.");
                            await output.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
                        }
                    }
                    Verify(partial, model.Hash);
                    File.Move(partial, path);
                }
                finally { if (File.Exists(partial)) File.Delete(partial); }
            }
            Verify(path, model.Hash);
            Console.WriteLine($"{model.Id}: verified {new FileInfo(path).Length} bytes");
        }
    }

    private static void Verify(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        if (!Convert.ToHexStringLower(SHA256.HashData(stream)).Equals(expected, StringComparison.Ordinal))
            throw new InvalidDataException("Checkpoint SHA-256 mismatch.");
    }
}
