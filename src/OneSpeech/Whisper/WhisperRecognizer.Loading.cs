using OneSpeech.Catalog;

namespace OneSpeech.Whisper;

public sealed partial class WhisperRecognizer
{
    /// <summary>Loads an already pulled whisper model without network access. Owns its graph resources; dispose after use. Options use the family defaults. Throws typed capability/model errors for the wrong family, I/O errors for missing files, and OperationCanceledException.</summary>
    public static WhisperRecognizer Load(ResolvedModel model, WhisperOptions? options = null, CancellationToken cancellationToken = default)
    {
        FamilyLoader.Require(model, "whisper");
        return Load(model.Directory, options, cancellationToken);
    }

    /// <summary>Pulls a pinned whisper checkpoint through the supplied store (ModelStore.Default when null), then loads CPU graphs. Model defaults to 'openai/whisper-large-v3-turbo'; options use family defaults. Downloads and conversion occur only when absent. Caller owns the result. Throws typed model/capability errors, I/O errors, or OperationCanceledException.</summary>
    public static async ValueTask<WhisperRecognizer> LoadAsync(ModelStore? store = null, string? model = null, WhisperOptions? options = null, CancellationToken cancellationToken = default)
    {
        var resolved = await FamilyLoader.ResolveAsync(store, model, "openai/whisper-large-v3-turbo", "whisper", cancellationToken).ConfigureAwait(false);
        return Load(resolved, options, cancellationToken);
    }
}
