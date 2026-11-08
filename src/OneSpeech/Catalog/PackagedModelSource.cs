namespace OneSpeech.Catalog;

// NuGet content is copied by the model package's transitive build target. Addressing
// files by catalog digest avoids registration, assembly loading and name collisions.
internal sealed class PackagedModelSource(ModelSource fallback) : ModelSource
{
    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) =>
        OpenReadAsync(model, file, null, cancellationToken);

    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, string? token, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (model.Tier == ModelTier.Tier1 && !model.Gated)
        {
            ModelCatalog.Validate(model);
            if (!model.Files.Contains(file)) throw new InvalidModelCatalogException("Packaged file is not a publisher file in this catalog entry.");
            string path = Path.Combine(AppContext.BaseDirectory, "onespeech-models", file.Sha256);
            if (File.Exists(path))
                return ValueTask.FromResult<Stream>(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, FileOptions.Asynchronous | FileOptions.SequentialScan));
        }
        return fallback.OpenReadAsync(model, file, token, cancellationToken);
    }
}
