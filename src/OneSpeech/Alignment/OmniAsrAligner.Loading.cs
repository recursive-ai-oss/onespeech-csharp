using OneSpeech.Catalog;

namespace OneSpeech.Alignment;

public sealed partial class OmniAsrAligner
{
    /// <summary>Loads an already pulled omniasr-ctc model without network access. Owns its graph resources; dispose after use. Threads default to 1. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static OmniAsrAligner Load(ResolvedModel model, int threads = 1, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "omniasr-ctc");
        return new OmniAsrAligner(Path.Combine(model.Directory, "ctc-fp32.onemodel"), Path.Combine(model.Directory, "omniASR_tokenizer.model"), model.Info, threads, cancellationToken);
    }

    /// <summary>Pulls a pinned omniasr-ctc checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'facebook/omniASR-CTC-300M'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<OmniAsrAligner> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "facebook/omniASR-CTC-300M", "omniasr-ctc", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
