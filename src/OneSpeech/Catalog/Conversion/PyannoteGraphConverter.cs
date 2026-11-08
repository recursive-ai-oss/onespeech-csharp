using static SpeakerGraph;

// Bredin & Laurent 2021 implementation-details section; Plaquet & Bredin 2023 §3.
// The published checkpoint supplies the analytic SincNet filter parameters and window.
internal sealed class PyannoteGraphConverter(Dictionary<string, TensorCheckpoint.Tensor> tensors)
{
    private readonly SpeakerGraph g = new();
    private readonly HashSet<string> used = new(StringComparer.Ordinal);
    private string W(string name)
    {
        if (!used.Add(name)) return name;
        var t = tensors[name];
        return g.Float(name, t.Shape, t.Values);
    }
    private string Transpose(string x, params long[] axes) => g.Op("Transpose", [x], Is("perm", axes));
    private string Activate(string x) => g.Op("LeakyRelu", [x], F("alpha", 0.01f));
    private string Norm(string x, string name) => g.Op("InstanceNormalization", [x, W(name + ".weight"), W(name + ".bias")], F("epsilon", 1e-5f));
    private string Linear(string x, string name) => g.Op("Add", [g.Op("MatMul", [x, Transpose(W(name + ".weight"), 1, 0)]), W(name + ".bias")]);

    internal void Save(string path)
    {
        string x = Norm("waveform", "sincnet.wav_norm1d");
        var low = tensors["sincnet.conv1d.0.filterbank.low_hz_"].Values;
        var band = tensors["sincnet.conv1d.0.filterbank.band_hz_"].Values;
        var window = tensors["sincnet.conv1d.0.filterbank.window_"].Values;
        var n = tensors["sincnet.conv1d.0.filterbank.n_"].Values;
        if (low.Length != 40 || band.Length != 40 || window.Length != 125 || n.Length != 125)
            throw new InvalidDataException("Unexpected analytic filterbank dimensions.");
        float[] filters = new float[80 * 251];
        for (int f = 0; f < 40; f++)
        {
            float lo = 50 + MathF.Abs(low[f]);
            float hi = Math.Clamp(lo + 50 + MathF.Abs(band[f]), 50, 8000);
            float width = hi - lo;
            for (int i = 0; i < 125; i++)
            {
                float a = lo * n[i], b = hi * n[i];
                float real = (MathF.Sin(b) - MathF.Sin(a)) / (n[i] / 2) * window[i] / (2 * width);
                float imaginary = (MathF.Cos(a) - MathF.Cos(b)) / (n[i] / 2) * window[i] / (2 * width);
                filters[f * 251 + i] = real;
                filters[f * 251 + 250 - i] = real;
                filters[(f + 40) * 251 + i] = imaginary;
                filters[(f + 40) * 251 + 250 - i] = -imaginary;
            }
            filters[f * 251 + 125] = 1;
        }
        x = g.Op("Conv", [x, g.Float("sincFilters", [80, 1, 251], filters)], Is("strides", 10));
        x = g.Op("Abs", [x]);
        for (int i = 0; i < 3; i++)
        {
            if (i != 0) x = g.Op("Conv", [x, W($"sincnet.conv1d.{i}.weight"), W($"sincnet.conv1d.{i}.bias")]);
            x = g.Op("MaxPool", [x], Is("kernel_shape", 3), Is("strides", 3));
            x = Activate(Norm(x, $"sincnet.norm1d.{i}"));
        }
        x = Transpose(x, 2, 0, 1);
        for (int layer = 0; layer < 4; layer++)
        {
            int inputs = layer == 0 ? 60 : 256;
            string Reorder(string stem, int columns, bool bias)
            {
                float[] result = new float[2 * (bias ? 8 : 4) * 128 * columns];
                for (int d = 0; d < 2; d++)
                {
                    string suffix = $"_l{layer}" + (d == 1 ? "_reverse" : "");
                    float[] source = tensors[$"lstm.{stem}{suffix}"].Values;
                    int[] order = [0, 3, 1, 2]; // PyTorch i,f,g,o → ONNX i,o,f,c.
                    for (int k = 0; k < 4; k++)
                        Array.Copy(source, order[k] * 128 * columns, result, (d * (bias ? 8 : 4) + k) * 128 * columns, 128 * columns);
                    if (bias)
                    {
                        source = tensors[$"lstm.bias_hh{suffix}"].Values;
                        for (int k = 0; k < 4; k++)
                            Array.Copy(source, order[k] * 128, result, d * 1024 + 512 + k * 128, 128);
                    }
                }
                return g.Float($"lstm{layer}{stem}", bias ? [2, 1024] : [2, 512, columns], result);
            }
            x = g.Op("LSTM", [x, Reorder("weight_ih", inputs, false), Reorder("weight_hh", 128, false), Reorder("bias_ih", 1, true)], I("hidden_size", 128), S("direction", "bidirectional"));
            x = g.Reshape(Transpose(x, 0, 2, 1, 3), 0, 1, 256);
        }
        x = Transpose(x, 1, 0, 2);
        x = Activate(Linear(x, "linear.0"));
        x = Activate(Linear(x, "linear.1"));
        x = g.Op("LogSoftmax", [Linear(x, "classifier")], I("axis", -1));
        g.Save(path, x, 7, inputName: "waveform", inputShape: [1, 1, 160000], outputName: "log_probabilities", outputShape: [1, 589, 7]);
    }
}
