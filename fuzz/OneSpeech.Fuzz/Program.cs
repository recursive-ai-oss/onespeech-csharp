using OneSpeech;
using SharpFuzz;

Fuzzer.LibFuzzer.Run(data =>
{
    try { _ = Transcript.FromJson(data); }
    catch (InvalidTranscriptJsonException) { }
});
