using OneSpeech.Catalog;

namespace OneSpeech.Embedders;

public sealed partial class CampPlusEmbedder
{
    /// <summary>Loads an already pulled campplus model without network access. Owns its graph resources; dispose after use. Threads default to 1. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static CampPlusEmbedder Load(ResolvedModel model, int threads = 1, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "speaker-campplus");
        bool chinese = model.Info.Id == "iic/speech_campplus_sv_zh-cn_16k-common";
        return new CampPlusEmbedder(Path.Combine(model.Directory, chinese ? "campplus-zh-fp32.onemodel" : "campplus-en-fp32.onemodel"), model.Info, chinese ? 192 : 512, threads, cancellationToken);
    }

    /// <summary>Pulls a pinned campplus checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'iic/speech_campplus_sv_en_voxceleb_16k'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<CampPlusEmbedder> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "iic/speech_campplus_sv_en_voxceleb_16k", "speaker-campplus", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
