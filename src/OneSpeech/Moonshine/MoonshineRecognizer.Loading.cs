using OneSpeech.Catalog;

namespace OneSpeech.Moonshine;

public sealed partial class MoonshineRecognizer
{
    /// <summary>Loads an already pulled moonshine model without network access. Owns its graph resources; dispose after use. Options use the family defaults. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static MoonshineRecognizer Load(ResolvedModel model, MoonshineOptions? options = null, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "moonshine");
        return Load(model.Directory, options, cancellationToken);
    }

    /// <summary>Pulls a pinned moonshine checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'moonshine-ai/moonshine-streaming-tiny'; options use family defaults. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<MoonshineRecognizer> LoadAsync(ModelStore? store = null, string? model = null, MoonshineOptions? options = null, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "moonshine-ai/moonshine-streaming-tiny", "moonshine", cancellationToken).ConfigureAwait(false);
        return Load(resolved, options, cancellationToken);
    }
}
