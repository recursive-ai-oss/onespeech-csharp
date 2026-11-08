using OneSpeech.Audio;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Qwen3;

internal sealed class QwenModelSession : IDisposable
{
    private readonly Session front;
    private readonly Session encoder;
    private readonly Session embedding;
    private readonly Session head;
    private readonly Session[] decoders;
    internal QwenConfiguration Configuration { get; }
    private readonly bool bfloat16;

    internal QwenModelSession(string directory, int threads, CancellationToken cancellationToken, Qwen3Precision precision = Qwen3Precision.Binary32)
    {
        bfloat16 = precision == Qwen3Precision.BFloat16;
        Configuration = QwenConfiguration.Load(directory);
        // Every autoregressive step changes the KV length, so an older shape is not reusable.
        // Retaining it also retains its activation slab until eviction.
        var options = new ModelOptions { IntraOpThreads = threads, MaxCachedPlans = 1 };
        var loaded = new List<Session>();
        Session Load(string name)
        {
            var result = Session.Load(Path.Combine(directory, name + (bfloat16 ? "-bf16" : "") + ".onemodel"), options, cancellationToken: cancellationToken);
            loaded.Add(result); return result;
        }
        try
        {
            front = Load("encoder-front"); encoder = Load("encoder"); embedding = Load("embeddings"); head = Load("head");
            decoders = new Session[(Configuration.Layers + 3) / 4];
            for (int i = 0; i < decoders.Length; i++) decoders[i] = Load("decoder-" + i);
        }
        catch { foreach (var session in loaded) session.Dispose(); throw; }
    }

    internal SessionOutputs Encode(AudioBuffer audio, CancellationToken cancellationToken)
    {
        float[] mel = new WhisperLogMelSpectrogram(128).ExtractQwen(audio, cancellationToken);
        if (bfloat16) for (int i = 0; i < mel.Length; i++) mel[i] = QwenRounding.BFloat16(mel[i]);
        int frames = mel.Length / 128, chunks = (frames + 99) / 100, width = Configuration.AudioWidth;
        float[] convolutionInput = new float[chunks * 128 * 100];
        for (int chunk = 0; chunk < chunks; chunk++)
            for (int bin = 0; bin < 128; bin++)
                mel.AsSpan(bin * frames + chunk * 100, Math.Min(100, frames - chunk * 100)).CopyTo(convolutionInput.AsSpan((chunk * 128 + bin) * 100));
        using var input = Tensor.FromSpan<float>(convolutionInput, Shape.Create(chunks, 1, 128, 100));
        using var convolved = front.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["mel_chunks"] = input });
        cancellationToken.ThrowIfCancellationRequested();
        var conv = convolved["conv"].Data<float>();
        // The released encoder resets sinusoidal positions at each 100-frame convolution chunk.
        int sequence = (chunks - 1) * 13 + (frames - (chunks - 1) * 100 + 7) / 8;
        float[] hidden = new float[sequence * width], positions = new float[sequence * width];
        int half = width / 2;
        for (int position = 0; position < sequence; position++)
            for (int dim = 0; dim < width; dim++)
            {
                float angle = position % 13 * MathF.Exp(-MathF.Log(10000) * (dim % half) / (half - 1));
                float positionValue = dim < half ? MathF.Sin(angle) : MathF.Cos(angle);
                if (bfloat16) positionValue = QwenRounding.BFloat16(positionValue);
                hidden[position * width + dim] = conv[position * width + dim];
                positions[position * width + dim] = positionValue;
            }
        float[] mask = new float[sequence * sequence];
        // The released CPU toolkit attends across the whole input despite emitting window metadata.
        // This is observed by perturbing frames beyond eight seconds and comparing encoder prefixes.
        using var hiddenTensor = Tensor.FromSpan<float>(hidden, Shape.Create(1, sequence, width));
        using var positionTensor = Tensor.FromSpan<float>(positions, Shape.Create(1, sequence, width));
        using var maskTensor = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, sequence, sequence));
        var result = encoder.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["hidden"] = hiddenTensor, ["positions"] = positionTensor, ["mask"] = maskTensor });
        if (cancellationToken.IsCancellationRequested) { result.Dispose(); cancellationToken.ThrowIfCancellationRequested(); }
        return result;
    }

    internal float[] Decode(ReadOnlySpan<int> tokens, Tensor? audio, ref SessionOutputs[]? past, int pastLength, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int length = tokens.Length, total = checked(pastLength + length), dim = Configuration.HeadDim;
        if (length <= 0 || pastLength < 0 || total > 32768) throw new ArgumentOutOfRangeException(nameof(tokens));
        long[] values = new long[length];
        for (int i = 0; i < length; i++) values[i] = tokens[i];
        using var tokenInput = Tensor.FromSpan<long>(values, Shape.Create(1, length));
        int audioCount = audio.HasValue ? audio.Value.Data<float>().Length / Configuration.Width : 0;
        long[] audioPositions = new long[audioCount * 2];
        int audioCursor = 0;
        for (int i = 0; i < length; i++)
            if (tokens[i] == 151676)
            {
                if (audioCursor >= audioCount) throw new InvalidDataException("More audio placeholders than encoder outputs.");
                audioPositions[audioCursor++ * 2 + 1] = i;
            }
        if (audioCursor != audioCount) throw new InvalidDataException("Audio placeholder count differs from encoder output length.");
        using var emptyAudio = Tensor.Create(DType.F32, Shape.Create(1, 0, Configuration.Width));
        using var audioPositionInput = Tensor.FromSpan<long>(audioPositions, Shape.Create(audioCount, 2));
        using var embedded = embedding.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["tokens"] = tokenInput, ["audio"] = audio ?? emptyAudio, ["audio_positions"] = audioPositionInput });
        Tensor hidden = embedded["embeddings"];
        float[] cos = new float[length * dim], sin = new float[length * dim], mask = new float[length * total];
        for (int i = 0; i < length; i++)
        {
            for (int d = 0; d < dim; d++)
            {
                float angle = (pastLength + i) * MathF.Pow(1000000, -(float)(2 * (d % (dim / 2))) / dim);
                cos[i * dim + d] = bfloat16 ? QwenRounding.BFloat16(MathF.Cos(angle)) : MathF.Cos(angle); sin[i * dim + d] = bfloat16 ? QwenRounding.BFloat16(MathF.Sin(angle)) : MathF.Sin(angle);
            }
            for (int k = pastLength + i + 1; k < total; k++) mask[i * total + k] = float.NegativeInfinity;
        }
        using var cosInput = Tensor.FromSpan<float>(cos, Shape.Create(1, 1, length, dim));
        using var sinInput = Tensor.FromSpan<float>(sin, Shape.Create(1, 1, length, dim));
        using var maskInput = Tensor.FromSpan<float>(mask, Shape.Create(1, 1, length, total));
        using var empty = Tensor.Create(DType.F32, Shape.Create(1, Configuration.KvHeads, 0, dim));
        var next = new SessionOutputs[decoders.Length];
        try
        {
            for (int part = 0; part < decoders.Length; part++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var inputs = new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["hidden"] = hidden, ["cos"] = cosInput, ["sin"] = sinInput, ["mask"] = maskInput };
                for (int layer = part * 4; layer < Math.Min(part * 4 + 4, Configuration.Layers); layer++)
                {
                    inputs["past_k_" + layer] = past == null ? empty : past[part]["present_k_" + layer];
                    inputs["past_v_" + layer] = past == null ? empty : past[part]["present_v_" + layer];
                }
                next[part] = decoders[part].Run(inputs); hidden = next[part]["decoder"];
            }
            using var logits = head.Run(new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["hidden"] = hidden });
            float[] result = logits["logits"].Data<float>().ToArray();
            if (past != null) foreach (var previous in past) previous.Dispose();
            past = next; return result;
        }
        catch { foreach (var item in next) item?.Dispose(); throw; }
    }

    public void Dispose()
    {
        foreach (var decoder in decoders) decoder.Dispose();
        head.Dispose(); embedding.Dispose(); encoder.Dispose(); front.Dispose();
    }
}
