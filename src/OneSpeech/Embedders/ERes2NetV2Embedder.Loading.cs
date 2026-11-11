using OneSpeech.Catalog;

namespace OneSpeech.Embedders;

public sealed partial class ERes2NetV2Embedder
{
    /// <summary>Loads an already pulled ERes2NetV2 model without network access. Owns its graph resources; dispose after use. Threads default to 1. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static ERes2NetV2Embedder Load(ResolvedModel model, int threads = 1, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "speaker-eres2netv2");
        return new ERes2NetV2Embedder(Path.Combine(model.Directory, "eres2netv2-fp32.onemodel"), model.Info, threads, cancellationToken);
    }

    /// <summary>Pulls the pinned ERes2NetV2 checkpoint through the supplied store (ModelStore.Default when null), then loads a CPU graph. Model defaults to 'iic/speech_eres2netv2_sv_zh-cn_16k-common'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<ERes2NetV2Embedder> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "iic/speech_eres2netv2_sv_zh-cn_16k-common", "speaker-eres2netv2", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
