if (args is ["speaker-extract-ecapa", var ecapaGraph, var ecapaData, var ecapaIds, var ecapaDestination, var ecapaStart, var ecapaCount])
{
    SpeakerBatch.Extract(ecapaGraph, ecapaData, ecapaIds, ecapaDestination, int.Parse(ecapaStart, System.Globalization.CultureInfo.InvariantCulture), int.Parse(ecapaCount, System.Globalization.CultureInfo.InvariantCulture), 192, ecapa: true);
    return;
}
if (args is ["speaker-probe-ecapa", var ecapaProbeGraph, var ecapaAudio, var ecapaProbeOutput])
{
    SpeakerProbe.RunEcapa(ecapaProbeGraph, ecapaAudio, ecapaProbeOutput);
    return;
}
if (args is ["speaker-convert-ecapa", var ecapaInput, var ecapaOutput])
{
    using var input = File.OpenRead(ecapaInput);
    new EcapaConverter(TensorCheckpoint.Read(input)).Save(ecapaOutput);
    return;
}
if (args is ["speaker-ids", var idsTrials, var idsClean, var idsOutput, var idsParity])
{
    SpeakerBatch.PrepareIds(idsTrials, idsClean, idsOutput, idsParity);
    return;
}
if (args is ["speaker-extract-camp", var extractGraph, var extractData, var extractIds, var extractOutput, var extractStart, var extractCount, var extractDimension])
{
    SpeakerBatch.Extract(extractGraph, extractData, extractIds, extractOutput, int.Parse(extractStart, System.Globalization.CultureInfo.InvariantCulture), int.Parse(extractCount, System.Globalization.CultureInfo.InvariantCulture), int.Parse(extractDimension, System.Globalization.CultureInfo.InvariantCulture));
    return;
}
if (args is ["speaker-run-features", var featureGraph, var featureInput, var vectorOutput])
{
    SpeakerProbe.RunFeatures(featureGraph, featureInput, vectorOutput);
    return;
}
if (args is ["speaker-trials", var cleanRoot, var otherRoot, var trialsOutput, var seed])
{
    SpeakerVerification.Generate(cleanRoot, otherRoot, trialsOutput, int.Parse(seed, System.Globalization.CultureInfo.InvariantCulture));
    return;
}
if (args is ["speaker-score", var trialFile, var oursDirectory, var referenceDirectory, var scoreOutput])
{
    SpeakerVerification.ScoreFiles(trialFile, oursDirectory, referenceDirectory, scoreOutput);
    return;
}
if (args is ["speaker-probe", var graphPath, var audioPath, var probeOutput])
{
    SpeakerProbe.Run(graphPath, audioPath, probeOutput);
    return;
}
if (args is ["speaker-convert-camp", var campCheckpoint, var campOutput])
{
    using var campInput = File.OpenRead(campCheckpoint);
    new CampPlusConverter(TensorCheckpoint.Read(campInput)).Save(campOutput);
    return;
}
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
