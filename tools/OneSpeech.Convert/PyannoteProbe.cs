using System.Buffers.Binary;
using System.Globalization;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

internal static class PyannoteProbe
{
    internal static void Run(string checkpoint, string cache, string probes)
    {
        Directory.CreateDirectory(cache);
        string graphPath = Path.Combine(cache, "segmentation.onemodel");
        using (var input = File.OpenRead(checkpoint)) new PyannoteGraphConverter(TensorCheckpoint.Read(input, segmentation: true)).Save(graphPath);
        using var session = Session.Load(graphPath, new ModelOptions { IntraOpThreads = 4 });
        foreach (string path in Directory.GetFiles(probes, "input-*.f32").Order(StringComparer.Ordinal))
        {
            byte[] bytes = File.ReadAllBytes(path);
            using var input = Tensor.Zeros(DType.F32, Shape.Create(1, 1, bytes.Length / 4));
            for (int i = 0; i < bytes.Length / 4; i++) input.MutableData<float>()[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
            using var output = session.Run(new Dictionary<string, Tensor> { ["waveform"] = input });
            var actual = output["log_probabilities"].Data<float>();
            string expectedPath = Path.Combine(probes, Path.GetFileName(path).Replace("input-", "output-", StringComparison.Ordinal));
            bytes = File.ReadAllBytes(expectedPath);
            if (bytes.Length / 4 != actual.Length) throw new InvalidDataException("Frame shape mismatch.");
            double sum = 0, max = 0, probabilitySum = 0, probabilityMax = 0;
            int decisionsDiffer = 0;
            for (int f = 0; f < actual.Length / 7; f++)
            {
                int a = 0, b = 0;
                for (int c = 1; c < 7; c++)
                {
                    if (actual[f * 7 + c] > actual[f * 7 + a]) a = c;
                    if (BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan((f * 7 + c) * 4, 4)) > BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan((f * 7 + b) * 4, 4))) b = c;
                }
                if (a != b) decisionsDiffer++;
            }
            for (int i = 0; i < actual.Length; i++)
            {
                double error = Math.Abs(actual[i] - BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4)));
                sum += error; max = Math.Max(max, error);
                double pError = Math.Abs(Math.Exp(actual[i]) - Math.Exp(BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4))));
                probabilitySum += pError; probabilityMax = Math.Max(probabilityMax, pError);
            }
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{Path.GetFileName(path)} count={actual.Length} logMax={max:G9} logMean={sum / actual.Length:G9} probabilityMax={probabilityMax:G9} probabilityMean={probabilitySum / actual.Length:G9} argmaxMismatches={decisionsDiffer}"));
        }
    }
}
