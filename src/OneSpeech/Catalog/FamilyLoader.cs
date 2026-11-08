namespace OneSpeech.Catalog;

internal static class FamilyLoader
{
    internal static void Require(ResolvedModel model, string family)
    {
        ArgumentNullException.ThrowIfNull(model);
        var entry = ModelCatalog.Default.Get(model.Info.Id);
        if (entry.Family != family) throw new CapabilityNotSupportedException($"Model '{model.Info.Id}' is not a '{family}' model.");
    }

    internal static async ValueTask<ResolvedModel> ResolveAsync(ModelStore? store, string? model, string defaultId, string family, CancellationToken cancellationToken)
    {
        store ??= ModelStore.Default;
        var entry = store.Catalog.Get(model ?? defaultId);
        if (entry.Family != family) throw new CapabilityNotSupportedException($"Model '{entry.Id}' is not a '{family}' model.");
        return await store.PullAsync(entry.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
    }
}
