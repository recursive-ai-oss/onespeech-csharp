using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Embedders;
using OneSpeech.Speakers;

namespace OneSpeech.Pipeline;

internal static class SpeakerServices
{
    private sealed record Entry(CampPlusEmbedder Embedder, InMemorySpeakerRegistry Registry) : IDisposable
    {
        public void Dispose() { Registry.Dispose(); Embedder.Dispose(); }
    }

    private static readonly SemaphoreSlim Gate = new(1, 1);
    private static readonly Dictionary<string, Entry> Registries = new(StringComparer.Ordinal);

    internal static void Clear(CancellationToken token)
    {
        Gate.Wait(token);
        try
        {
            foreach (var entry in Registries.Values) entry.Dispose();
            Registries.Clear();
        }
        finally { Gate.Release(); }
    }

    internal static SpeakerMatch Match(string model, ReadOnlyMemory<float> embedding, CancellationToken token)
    {
        Gate.Wait(token);
        try { return Registries.TryGetValue(model, out var entry) ? entry.Registry.IdentifyEmbedding(embedding, token) : new SpeakerMatch { Score = double.NegativeInfinity }; }
        finally { Gate.Release(); }
    }

    internal static void Enroll(string name, string path, string? language, string? model, CancellationToken token)
    {
        language = LanguageTag.Normalize(language);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        RunAsync(name, path, null, language, model, token).AsTask().GetAwaiter().GetResult();
    }
    internal static SpeakerMatch Identify(string path, string? language, string? model, CancellationToken token)
        => RunAsync(null, path, null, language, model, token).AsTask().GetAwaiter().GetResult();
    internal static double Verify(string first, string second, string? language, string? model, CancellationToken token)
        => RunAsync(null, first, second, language, model, token).AsTask().GetAwaiter().GetResult().Score;

    internal static async ValueTask EnrollAsync(string name, Stream stream, string? language, string? model, CancellationToken token)
    {
        language = LanguageTag.Normalize(language);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        await SpoolAsync(stream, path => RunAsync(name, path, null, language, model, token), token).ConfigureAwait(false);
    }
    internal static ValueTask<SpeakerMatch> IdentifyAsync(Stream stream, string? language, string? model, CancellationToken token)
        => SpoolAsync(stream, path => RunAsync(null, path, null, language, model, token), token);
    internal static async ValueTask<double> VerifyAsync(Stream first, Stream second, string? language, string? model, CancellationToken token)
        => (await SpoolAsync(first, a => SpoolAsync(second, b => RunAsync(null, a, b, language, model, token), token), token).ConfigureAwait(false)).Score;

    private static async ValueTask<SpeakerMatch> SpoolAsync(Stream stream, Func<string, ValueTask<SpeakerMatch>> action, CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead) throw new ArgumentException("A readable encoded-audio stream is required.", nameof(stream));
        token.ThrowIfCancellationRequested();
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        await using var file = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.Read, 65536, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
        await stream.CopyToAsync(file, token).ConfigureAwait(false);
        await file.FlushAsync(token).ConfigureAwait(false);
        return await action(path).ConfigureAwait(false);
    }

    internal static async ValueTask EnrollAsync(string name, string path, string? language, string? model, CancellationToken token) =>
        _ = await RunAsync(name, path, null, language, model, token).ConfigureAwait(false);
    internal static ValueTask<SpeakerMatch> IdentifyAsync(string path, string? language, string? model, CancellationToken token) =>
        RunAsync(null, path, null, language, model, token);
    internal static async ValueTask<double> VerifyAsync(string first, string second, string? language, string? model, CancellationToken token) =>
        (await RunAsync(null, first, second, language, model, token).ConfigureAwait(false)).Score;

    private static async ValueTask<SpeakerMatch> RunAsync(string? name, string path, string? second, string? language, string? model, CancellationToken token)
    {
        language = LanguageTag.Normalize(language);
        if (name is not null) ArgumentException.ThrowIfNullOrWhiteSpace(name);
        string id = model ?? SpeechServices.EmbedderId(language);
        if (id != SpeechServices.EmbedderId(null) && id != SpeechServices.EmbedderId("zh"))
            throw new CapabilityNotSupportedException("Speaker recognition requires a registered CAM++ English or Chinese embedder model ID.");
        // Decode before loading models; a failed enrollment never replaces an existing name.
        var audio = Read(path, token);
        var other = second is null ? null : Read(second, token);
        await Gate.WaitAsync(token).ConfigureAwait(false);
        try
        {
            if (!Registries.TryGetValue(id, out var entry))
            {
                var resolved = await ModelStore.Default.PullAsync(id, cancellationToken: token).ConfigureAwait(false);
                bool chinese = id == SpeechServices.EmbedderId("zh");
                var embedder = new CampPlusEmbedder(Path.Combine(resolved.Directory, chinese ? "campplus-zh-fp32.onemodel" : "campplus-en-fp32.onemodel"), resolved.Info, chinese ? 192 : 512, cancellationToken: token);
                entry = new Entry(embedder, new InMemorySpeakerRegistry(embedder, chinese ? ChineseThreshold : EnglishThreshold));
                Registries.Add(id, entry);
            }
            if (name is not null) { entry.Registry.Enroll(name, audio, token); return new SpeakerMatch { SpeakerId = name, Score = 1 }; }
            return other is null ? entry.Registry.Identify(audio, token) : new SpeakerMatch { Score = entry.Registry.Verify(audio, other, token) };
        }
        finally { Gate.Release(); }
    }

    // Inclusive raw-cosine operating points fitted on the development trials documented in speakers.md.
    internal const double EnglishThreshold = 0.4929986257373565;
    internal const double ChineseThreshold = 0.7229075438038025;

    private static AudioBuffer Read(string path, CancellationToken token)
    {
        using var source = AudioFileSource.Open(path, token);
        var samples = new List<float>();
        var block = new float[4096 * source.Channels];
        int frames;
        while ((frames = source.Read(block, token)) != 0)
        {
            if (samples.Count + frames > 60 * 16000) throw new CapabilityNotSupportedException("Speaker samples are limited to 60 seconds per input; select a speech excerpt.");
            for (int i = 0; i < frames; i++)
            {
                double sum = 0;
                for (int c = 0; c < source.Channels; c++) sum += block[i * source.Channels + c];
                samples.Add((float)(sum / source.Channels));
            }
        }
        if (samples.Count == 0) throw new UnsupportedAudioException("Speaker samples must contain nonempty speech audio.");
        return new AudioBuffer(samples.ToArray(), 16000);
    }
}
