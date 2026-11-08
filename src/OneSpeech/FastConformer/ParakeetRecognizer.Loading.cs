using OneSpeech.Catalog;

namespace OneSpeech.FastConformer;

public sealed partial class ParakeetRecognizer
{

    /// <summary>Pulls a pinned parakeet-tdt checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'nvidia/parakeet-tdt-0.6b-v3'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<ParakeetRecognizer> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
