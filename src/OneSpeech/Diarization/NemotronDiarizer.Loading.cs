using OneSpeech.Catalog;

namespace OneSpeech.Diarization;

public sealed partial class NemotronDiarizer
{

    /// <summary>Pulls a pinned nemotron-diarization checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'nvidia/Nemotron-3-Diarization'; threads default to 1. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<NemotronDiarizer> LoadAsync(ModelStore? store = null, string? model = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "nvidia/Nemotron-3-Diarization", "nemotron-diarization", cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }
}
