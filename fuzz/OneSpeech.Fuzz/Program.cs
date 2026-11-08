using OneSpeech;
using OneSpeech.Catalog;
using SharpFuzz;

bool catalog = Environment.GetEnvironmentVariable("ONESPEECH_FUZZ_CATALOG") == "1";
Fuzzer.LibFuzzer.Run(data =>
{
    if (catalog)
    {
        try { _ = ModelCatalog.FromJson(data); }
        catch (InvalidModelCatalogException) { }
    }
    else
    {
        try { _ = Transcript.FromJson(data); }
        catch (InvalidTranscriptJsonException) { }
    }
});
