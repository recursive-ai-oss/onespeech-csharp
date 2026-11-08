using System.Collections.Immutable;
using OneSpeech;
using OneSpeech.Catalog;

internal static class OmniSmoke
{
    internal static void Run()
    {
        string? root = Environment.GetEnvironmentVariable("ONESPEECH_OMNI_WEIGHTS");
        if (root is null) return;
        string fixture = Environment.GetEnvironmentVariable("ONESPEECH_OMNI_FIXTURE") ?? throw new InvalidOperationException("Set ONESPEECH_OMNI_FIXTURE.");
        string cache = Environment.GetEnvironmentVariable("ONESPEECH_OMNI_CACHE") ?? throw new InvalidOperationException("Set ONESPEECH_OMNI_CACHE outside the checkout.");
        var previous = ModelStore.Default;
        ModelStore.Default = new ModelStore(cache, new FlatSource(root));
        try
        {
            const string text = "Stuff it into you, his belly counselled him.";
            var result = Speech.Align(fixture, text, language: "en");
            if (result.Text != text || result.Words.Length != 8 || !result.Words.Any(w => w.TimingMethod == TimingMethod.CtcAligner && w.Start is not null))
                throw new InvalidOperationException("CTC public file alignment failed.");
            Console.WriteLine($"CTC fixture: {result.Words.Count(w => w.Start is not null && w.End is not null)}/{result.Words.Length} written words timed.");
            var written = new Transcript { Text = text, Words = result.Words.Select(w => w with { Start = null, End = null, TimingMethod = TimingMethod.None, Confidence = 0.3 }).ToImmutableArray() };
            var aligned = Speech.Align(fixture, written, language: "en");
            if (!aligned.Words.Zip(written.Words).All(pair => (pair.First with { Start = null, End = null, TimingMethod = TimingMethod.None }) == pair.Second))
                throw new InvalidOperationException("CTC transcript overload changed written words.");
            using (var encoded = new FileStream(fixture, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous))
            {
                var streamed = Speech.AlignAsync(encoded, text, language: "en", cancellationToken: CancellationToken.None).AsTask().GetAwaiter().GetResult();
                if (!encoded.CanRead || streamed.Text != text || !streamed.Words.Any(w => w.Start is not null))
                    throw new InvalidOperationException("CTC encoded-stream alignment or ownership failed.");
            }
            string silent = Path.Combine(Path.GetTempPath(), "ctc-silence-" + Guid.NewGuid().ToString("N") + ".wav");
            try
            {
                using (var wave = new BinaryWriter(File.Create(silent)))
                {
                    wave.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); wave.Write(32036);
                    wave.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); wave.Write(16); wave.Write((short)1); wave.Write((short)1);
                    wave.Write(16000); wave.Write(32000); wave.Write((short)2); wave.Write((short)16);
                    wave.Write(System.Text.Encoding.ASCII.GetBytes("data")); wave.Write(32000); wave.Write(new byte[32000]);
                }
                if (Speech.Align(silent, "Hello", language: "en").Words.Any(w => w.Start is not null || w.End is not null))
                    throw new InvalidOperationException("CTC invented a word timing on silence.");
            }
            finally { File.Delete(silent); }
            Console.WriteLine("CTC first-load conversion, public file/stream/transcript alignment, written-word preservation and silence rejection passed.");
        }
        finally { ModelStore.Default = previous; }
    }

    private sealed class FlatSource(string root) : ModelSource
    {
        public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<Stream>(new FileStream(Path.Combine(root, file.Name), FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous));
        }
    }
}
