using OneSpeech.Evaluation;
using SharpFuzz;

Fuzzer.LibFuzzer.Run(data =>
{
    using var input = new MemoryStream(data.ToArray());
    try { _ = Rttm.Read(input); }
    catch (InvalidDataException) { }
});
