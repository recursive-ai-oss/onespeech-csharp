if (args is ["speaker-fetch", var root])
{
    await SpeakerTools.FetchAsync(root, CancellationToken.None).ConfigureAwait(false);
    return;
}
if (args is ["speaker-inspect", var checkpoint])
{
    using var input = File.OpenRead(checkpoint);
    foreach (var pair in TensorCheckpoint.Read(input))
        Console.WriteLine($"{pair.Key} [{string.Join(',', pair.Value.Shape)}]");
    return;
}
Console.WriteLine("In development.");
