using OneSpeech.Catalog;

namespace OneSpeech.Embedders;

public sealed partial class EcapaTdnnEmbedder
{
    /// <summary>Loads an already pulled ecapa-tdnn model without network access. Owns its graph resources; dispose after use. Threads default to 1. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static EcapaTdnnEmbedder Load(ResolvedModel model, int threads = 1, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "speaker-ecapa");
        return new EcapaTdnnEmbedder(Path.Combine(model.Directory, "ecapa-fp32.onemodel"), model.Info, threads, cancellationToken);
    }

    /// <summary>Pulls a pinned ecapa-tdnn checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'speechbrain/spkrec-ecapa-voxceleb'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<EcapaTdnnEmbedder> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "speechbrain/spkrec-ecapa-voxceleb", "speaker-ecapa", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
