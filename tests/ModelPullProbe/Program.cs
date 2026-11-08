using OneSpeech.Catalog;

if (args is not [var cache]) throw new ArgumentException("Pass a disposable cache root.");
var source = new CountingSource();
var store = new ModelStore(cache, source);
var first = await store.PullAsync("openai/whisper-tiny");
int afterFirst = source.Calls;
if (afterFirst != ModelCatalog.Default.Get("openai/whisper-tiny").Files.Length) throw new InvalidOperationException("Use a fresh cache for the first pull.");
var second = await store.PullAsync("openai/whisper-tiny");
if (source.Calls != afterFirst || first.Directory != second.Directory) throw new InvalidOperationException("Cache hit made a request.");
// A new store must rehash disk bytes, still without fetching.
await new ModelStore(cache, new RejectingSource()).PullAsync("openai/whisper-tiny");
Console.WriteLine($"first_source_opens={afterFirst}; second_source_opens={source.Calls - afterFirst}; fresh_store_source_opens=0; verified_files={first.Info.Files.Length}");
Console.WriteLine(first.Directory);

internal sealed class CountingSource : ModelSource
{
    private readonly HttpModelSource http = new();
    internal int Calls { get; private set; }
    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken)
    {
        Calls++;
        return http.OpenReadAsync(model, file, cancellationToken);
    }
}
internal sealed class RejectingSource : ModelSource
{
    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => throw new InvalidOperationException("Unexpected fetch on cache hit.");
}
