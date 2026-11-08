using OneSpeech.Catalog;
using OneSpeech.Qwen3;

namespace OneSpeech.Pipeline;

internal static partial class SpeechServices
{
    internal static async ValueTask PrefetchAsync(string? language, string? model, CancellationToken cancellationToken)
    {
        language = LanguageTag.Normalize(language);
        var store = ModelStore.Default;
        foreach (string id in PrefetchModels(store, language, model))
        {
            cancellationToken.ThrowIfCancellationRequested();
            _ = await store.PullAsync(id, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
    }

    internal static string[] PrefetchModels(ModelStore store, string? language, string? model)
    {
        language = LanguageTag.Normalize(language);
        var entry = model is null ? store.Catalog.GetRoleDefault("transcription") : store.Catalog.Get(model);
        bool recognizer = entry.Family is "whisper" or "moonshine" or "qwen3-asr" or "parakeet-tdt" or "nemotron-asr";
        if (recognizer || entry.Capabilities.ForcedAlignment) language = LanguageTag.Require(language, entry.Capabilities, entry.Id);
        var ids = new List<string> { entry.Id };
        if (recognizer)
        {
            if (!entry.Capabilities.NativeTimestamps || entry.Family == "qwen3-asr")
            {
                if (language is null || Qwen3ForcedAligner.SupportedLanguages.Contains(language)) ids.Add("Qwen/Qwen3-ForcedAligner-0.6B");
                ids.Add("facebook/omniASR-CTC-300M");
            }
            ids.Add(EmbedderId(language));
            if (language is null) ids.Add(EmbedderId("zh"));
        }
        return ids.Distinct(StringComparer.Ordinal).ToArray();
    }
}
