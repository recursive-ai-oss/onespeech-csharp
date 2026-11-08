using OneSpeech.Catalog;

namespace OneSpeech.Qwen3;

public sealed partial class Qwen3ForcedAligner
{
    /// <summary>Loads an already pulled qwen3-aligner model without network access. Owns its graph resources; dispose after use. Threads default to 1. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static Qwen3ForcedAligner Load(ResolvedModel model, int threads = 1, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "qwen3-aligner");
        return Load(model.Directory, threads, cancellationToken);
    }

    /// <summary>Pulls a pinned qwen3-aligner checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'Qwen/Qwen3-ForcedAligner-0.6B'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<Qwen3ForcedAligner> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "Qwen/Qwen3-ForcedAligner-0.6B", "qwen3-aligner", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
