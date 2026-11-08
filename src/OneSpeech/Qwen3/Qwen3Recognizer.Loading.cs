using OneSpeech.Catalog;

namespace OneSpeech.Qwen3;

public sealed partial class Qwen3Recognizer
{
    /// <summary>Loads an already pulled qwen3 model without network access. Owns its graph resources; dispose after use. Options use the family defaults. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static Qwen3Recognizer Load(ResolvedModel model, Qwen3Options? options = null, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "qwen3-asr");
        return Load(model.Directory, options, cancellationToken);
    }

    /// <summary>Pulls a pinned qwen3 checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'Qwen/Qwen3-ASR-0.6B'; options use family defaults. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<Qwen3Recognizer> LoadAsync(ModelStore? store = null, string? model = null, Qwen3Options? options = null, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "Qwen/Qwen3-ASR-0.6B", "qwen3-asr", cancellationToken).ConfigureAwait(false);
        return Load(resolved, options, cancellationToken);
    }
}
