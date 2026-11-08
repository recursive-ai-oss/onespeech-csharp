using OneSpeech;
using OneSpeech.Catalog;

try
{
    if (args is ["--version"]) Console.WriteLine(BuildVersion.Value);
    else if (args is ["models", "list"])
    {
        foreach (var model in ModelCatalog.Default.Entries)
            Console.WriteLine($"{model.Id}\t{model.Tier}\t{model.License}\t{model.Revision}");
    }
    else if (args is ["models", "pull", var id])
    {
        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
        var result = await new ModelStore().PullAsync(id, cancellationToken: cancellation.Token);
        Console.WriteLine(result.Directory);
    }
    else
    {
        Console.WriteLine("Speech recognition for .NET.\nUsage: onespeech models list | models pull <id> | --version");
        return 2;
    }
    return 0;
}
catch (OneSpeechException e) { Console.Error.WriteLine(e.Message); return 1; }
catch (OperationCanceledException) { Console.Error.WriteLine("Model pull cancelled."); return 130; }
