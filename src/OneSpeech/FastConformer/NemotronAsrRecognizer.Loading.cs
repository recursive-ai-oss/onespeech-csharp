using OneSpeech.Catalog;

namespace OneSpeech.FastConformer;

public sealed partial class NemotronAsrRecognizer
{

    /// <summary>Pulls a pinned nemotron-asr checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'nvidia/nemotron-3.5-asr-streaming-0.6b'; options use family defaults. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<NemotronAsrRecognizer> LoadAsync(ModelStore? store = null, string? model = null, NemotronAsrOptions? options = null, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "nvidia/nemotron-3.5-asr-streaming-0.6b", "nemotron-asr", cancellationToken).ConfigureAwait(false);
        return Load(resolved, options, cancellationToken);
    }
}
