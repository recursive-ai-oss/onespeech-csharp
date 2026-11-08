using OneSpeech.Audio;
using OneSpeech.Catalog;

namespace OneSpeech.Pipeline;

internal static partial class SpeechServices
{
    internal static async IAsyncEnumerable<Segment> TranscribeStream(Stream stream, string? language, string? model, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await using var spool = await SpoolAsync(stream, cancellationToken).ConfigureAwait(false);
        await foreach (var segment in TranscribeStream(spool.Name, language, model, cancellationToken).ConfigureAwait(false)) yield return segment;
    }

    internal sealed class EncodedSpool(string name) : IAsyncDisposable
    {
        internal string Name { get; } = name;
        public ValueTask DisposeAsync() { File.Delete(Name); return ValueTask.CompletedTask; }
    }

    internal static async ValueTask<EncodedSpool> SpoolAsync(Stream stream, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead) throw new ArgumentException("Input stream must be readable.", nameof(stream));
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        var spool = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, FileOptions.Asynchronous);
        try
        {
            byte[] buffer = new byte[65536];
            long total = 0; int count;
            while ((count = await stream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) != 0)
            {
                if ((total += count) > 512L * 1024 * 1024) throw new UnsupportedAudioException("Encoded stream exceeds the 512 MiB spool limit.");
                await spool.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
            }
            await spool.FlushAsync(cancellationToken).ConfigureAwait(false);
            // Close the writer before replay; reopening DeleteOnClose/write-shared handles is not portable.
            await spool.DisposeAsync().ConfigureAwait(false);
            return new EncodedSpool(path);
        }
        catch { await spool.DisposeAsync().ConfigureAwait(false); File.Delete(path); throw; }
    }

    internal static async ValueTask<string> DetectLanguageAsync(Stream stream, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        await using var spool = await SpoolAsync(stream, cancellationToken).ConfigureAwait(false);
        return await DetectLanguageAsync(spool.Name, language, model, cancellationToken).ConfigureAwait(false);
    }

    internal static async ValueTask<Transcript> AlignPathAsync(string path, string? text, Transcript? transcript, string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        if (text is null && transcript is null) throw new ArgumentNullException(nameof(transcript));
        using var source = AudioFileSource.Open(path, cancellationToken);
        var audio = ReadAlignmentAudio(source, cancellationToken);
        await PrepareAlignmentAsync(language, model, cancellationToken).ConfigureAwait(false);
        using var aligner = await ResolvePublicAlignerAsync(ModelStore.Default, model, cancellationToken).ConfigureAwait(false);
        return transcript is null ? aligner.Align(audio, text!, language, cancellationToken) : aligner.Align(audio, transcript, language, cancellationToken);
    }

    private static async ValueTask PrepareAlignmentAsync(string? language, string? model, CancellationToken cancellationToken)
    {
        var store = ModelStore.Default;
        if (model is not null)
        {
            var entry = store.Catalog.Get(model);
            if (!entry.Capabilities.ForcedAlignment) throw new CapabilityNotSupportedException($"Model '{model}' is not a forced aligner.");
            _ = LanguageTag.Require(language, entry.Capabilities, model);
            _ = await store.PullAsync(model, cancellationToken: cancellationToken).ConfigureAwait(false);
            return;
        }
        using var ordered = new OrderedTimingAligner(store);
        language = LanguageTag.Require(language, ordered.Capabilities, "ordered timing");
        if (language is not null && Qwen3.Qwen3ForcedAligner.SupportedLanguages.Contains(language))
            _ = await store.PullAsync("Qwen/Qwen3-ForcedAligner-0.6B", cancellationToken: cancellationToken).ConfigureAwait(false);
        _ = await store.PullAsync("facebook/omniASR-CTC-300M", cancellationToken: cancellationToken).ConfigureAwait(false);
    }
}
