using System.Collections.Immutable;
using System.Runtime.InteropServices;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Moonshine;

// Session-owned buffers; all shape variants are warmed before accepting audio.
internal sealed class MoonshineLiveWorkspace : IDisposable
{
    private const int MaxFrames = 1536; // More than the shared 30 s utterance horizon, including VAD frame rounding.
    private readonly MoonshineRecognizer model;
    private readonly MoonshineModelSession runtime;
    private readonly MoonshineConfiguration c;
    private readonly List<Tensor> owned = [];
    private readonly Tensor[] encoderInputs, encoderOutputs, stepOutputs;
    private readonly string[] projectNames, stepNames, stepOutputNames;
    private readonly Bucket[] buckets;
    private readonly Tensor token, cosine, sine, selfMask, index;
    private readonly Tensor cache;
    private readonly int cacheCapacity;
    private readonly float[] prefix, ropeCos, ropeSin;
    private readonly int[] ids;
    private readonly char[] text;
    private readonly byte[] bytes;
    private readonly int window, left;
    private int frozen, previous;

    internal MoonshineLiveWorkspace(MoonshineRecognizer model)
    {
        this.model = model; runtime = model.Runtime; c = runtime.Configuration;
        try
        {
            int capacity = 1;
            while (capacity < Math.Ceiling(31 * model.TokenRate) + 1) capacity *= 2;
            cacheCapacity = capacity;
            ids = new int[capacity]; bytes = new byte[capacity]; text = new char[checked(capacity * model.Tokenizer.MaximumPieceCharacters)];
            ropeCos = new float[capacity * c.RotaryDimension]; ropeSin = new float[ropeCos.Length];
            for (int p = 0; p < capacity; p++) for (int d = 0; d < c.RotaryDimension; d++)
            {
                float angle = p * (float)Math.Pow(10000, -2d * (d / 2) / c.RotaryDimension);
                ropeCos[p * c.RotaryDimension + d] = MathF.Cos(angle); ropeSin[p * c.RotaryDimension + d] = MathF.Sin(angle);
            }
            left = c.EncoderLayers * 15 + 3;
            window = (left + 12 + 16 + 15) / 16 * 16;
            prefix = new float[MaxFrames * c.EncoderWidth];
            encoderInputs = [Float(1, 1, window * 320), Float(1, 1, window, window), Float(1, 1, window, window)];
            encoderOutputs = [Float(1, window, c.EncoderWidth)];
            token = Long(1, 1); token.MutableData<long>()[0] = 1;
            cosine = Float(1, 1, 1, c.RotaryDimension); sine = Float(1, 1, 1, c.RotaryDimension);
            selfMask = Float(1, 1, 1, capacity); selfMask.MutableData<float>().Fill(float.NegativeInfinity); selfMask.MutableData<float>()[0] = 0;
            index = Long(1, c.DecoderHeads, c.DecoderHeadDimension);
            cache = Float(c.DecoderLayers * 2, capacity, c.DecoderHeads, c.DecoderHeadDimension);
            projectNames = ["cross"]; stepNames = ["tokens", "cos", "sin", "self_mask", "cross_mask", "cache_index", "past", "cross"];
            stepOutputNames = ["logits", "new_cache"];
            stepOutputs = [Float(1, 1, 32768), Float(c.DecoderLayers * 2, c.DecoderHeads, c.DecoderHeadDimension)];
            int[] sizes = [128, 256, 512, 1024, MaxFrames]; buckets = new Bucket[sizes.Length];
            for (int b = 0; b < sizes.Length; b++)
            {
                int frames = sizes[b];
                var memory = Float(1, frames, c.EncoderWidth); var positions = Long(frames);
                for (int i = 0; i < frames; i++) positions.MutableData<long>()[i] = i;
                var cross = new Tensor[projectNames.Length]; var inputs = new Tensor[stepNames.Length];
                var mask = Float(1, 1, 1, frames);
                inputs[0] = token; inputs[1] = cosine; inputs[2] = sine; inputs[3] = selfMask; inputs[4] = mask; inputs[5] = index;
                cross[0] = Float(c.DecoderLayers * 2, c.DecoderHeads, frames, c.DecoderHeadDimension);
                inputs[6] = cache; inputs[7] = cross[0];
                buckets[b] = new Bucket(frames, memory, [memory, positions], cross, mask, inputs);
            }
            SetMasks(window); runtime.RunLiveEncoder(encoderInputs, encoderOutputs);
            for (int b = 0; b < buckets.Length; b++)
            {
                var bucket = buckets[b]; runtime.RunProject(bucket.ProjectInputs, projectNames, bucket.Cross);
                runtime.RunStep(stepNames, bucket.StepInputs, stepOutputNames, stepOutputs);
            }
        }
        catch { foreach (var tensor in owned) tensor.Dispose(); owned.Clear(); throw; }
    }

    internal Segment Decode(ReadOnlyMemory<float> audio, bool final, TimeSpan start, TimeSpan end, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        int frames = (audio.Length + 319) / 320;
        if (frames == 0) return new Segment { Text = "", Start = start, End = end, Language = "en" };
        if (frames > MaxFrames) throw new UnsupportedAudioException("The live Moonshine workspace needs the shared 30-second VAD horizon.");
        if (frames < previous) frozen = 0;
        previous = frames;
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            int begin = Math.Max(0, frozen - left), count = Math.Min(audio.Length - begin * 320, window * 320);
            int valid = (count + 319) / 320;
            var samples = encoderInputs[0].MutableData<float>(); samples.Clear();
            // The released model masks the entire incomplete 80-sample frame.
            audio.Span.Slice(begin * 320, count / 80 * 80).CopyTo(samples);
            SetMasks(valid); runtime.RunLiveEncoder(encoderInputs, encoderOutputs);
            int relative = frozen - begin;
            encoderOutputs[0].Data<float>().Slice(relative * c.EncoderWidth, (valid - relative) * c.EncoderWidth).CopyTo(prefix.AsSpan(frozen * c.EncoderWidth));
            if (begin + valid == frames) { frozen = final ? frames : Math.Max(frozen, frames - 12); break; }
            frozen = begin + valid - 12;
        }
        Bucket bucket = buckets[0];
        for (int b = 0; b < buckets.Length; b++) if (frames <= buckets[b].Frames) { bucket = buckets[b]; break; }
        prefix.AsSpan(0, frames * c.EncoderWidth).CopyTo(bucket.Memory.MutableData<float>());
        var crossMask = bucket.Mask.MutableData<float>(); crossMask.Fill(float.NegativeInfinity); crossMask[..frames].Clear();
        runtime.RunProject(bucket.ProjectInputs, projectNames, bucket.Cross);
        int maximum = Math.Max(1, (int)(audio.Length / 16000d * model.TokenRate));
        if (maximum >= ids.Length) throw new UnsupportedAudioException("The live Moonshine token budget exceeded its 30-second workspace.");
        ids[0] = 1; int used = 1;
        var mask = selfMask.MutableData<float>(); mask.Fill(float.NegativeInfinity);
        for (int position = 0; position < maximum; position++)
        {
            ct.ThrowIfCancellationRequested();
            token.MutableData<long>()[0] = ids[position]; index.MutableData<long>().Fill(position); mask[position] = 0;
            ropeCos.AsSpan(position * c.RotaryDimension, c.RotaryDimension).CopyTo(cosine.MutableData<float>());
            ropeSin.AsSpan(position * c.RotaryDimension, c.RotaryDimension).CopyTo(sine.MutableData<float>());
            runtime.RunStep(stepNames, bucket.StepInputs, stepOutputNames, stepOutputs);
            var updated = stepOutputs[1].Data<float>(); var stored = cache.MutableData<float>();
            for (int plane = 0; plane < c.DecoderLayers * 2; plane++)
                updated.Slice(plane * c.Width, c.Width).CopyTo(stored.Slice((plane * cacheCapacity + position) * c.Width, c.Width));
            var logits = stepOutputs[0].Data<float>(); int best = 0;
            for (int id = 0; id < logits.Length; id++) { if (!float.IsFinite(logits[id])) throw new UnsupportedAudioException("Moonshine produced nonfinite logits."); if (logits[id] > logits[best]) best = id; }
            ids[used++] = best; if (best == 2) break;
        }
        model.TokenTrace?.Invoke(ids.AsSpan(0, used).ToArray());
        ct.ThrowIfCancellationRequested();
        int length = model.Tokenizer.DecodeInto(ids.AsSpan(0, used), text, bytes);
        string written = new(text.AsSpan(0, length));
        int wordCount = 0; bool inside = false;
        foreach (char ch in written) { if (ch == ' ') inside = false; else if (!inside) { wordCount++; inside = true; } }
        Word[] words = wordCount == 0 ? [] : new Word[wordCount]; int word = 0, offset = 0;
        while (offset < length)
        {
            while (offset < length && written[offset] == ' ') offset++;
            int first = offset; while (offset < length && written[offset] != ' ') offset++;
            if (offset > first) words[word++] = new Word { Text = new string(written.AsSpan(first, offset - first)), Language = "en" };
        }
        // Ownership transfers to the immutable event; this array is never retained or mutated.
        return new Segment { Text = written, Start = start, End = end, Language = "en", Words = words.Length == 0 ? [] : ImmutableCollectionsMarshal.AsImmutableArray(words) };
    }

    private void SetMasks(int valid)
    {
        var local = encoderInputs[1].MutableData<float>(); var causal = encoderInputs[2].MutableData<float>();
        for (int q = 0; q < window; q++) for (int k = 0; k < window; k++)
        {
            bool shared = q < valid ? k < valid && k >= q - 15 : k == 0;
            local[q * window + k] = shared && (q >= valid || k <= q + 3) ? 0 : float.NegativeInfinity;
            causal[q * window + k] = shared && (q >= valid || k <= q) ? 0 : float.NegativeInfinity;
        }
    }
    private Tensor Float(params int[] shape) { var tensor = Tensor.Zeros(DType.F32, Shape.Create(shape)); owned.Add(tensor); return tensor; }
    private Tensor Long(params int[] shape) { var tensor = Tensor.Zeros(DType.I64, Shape.Create(shape)); owned.Add(tensor); return tensor; }
    internal void InspectRuns(Action<string, Action> inspect)
    {
        inspect("encoder", () => runtime.RunLiveEncoder(encoderInputs, encoderOutputs));
        inspect("project", () => runtime.RunProject(buckets[0].ProjectInputs, projectNames, buckets[0].Cross));
        inspect("step", () => runtime.RunStep(stepNames, buckets[0].StepInputs, stepOutputNames, stepOutputs));
        inspect("tokenizer", () => model.Tokenizer.DecodeInto(ids.AsSpan(0, 1), text, bytes));
    }
    internal void Reset() { frozen = 0; previous = 0; }
    public void Dispose() { foreach (var tensor in owned) tensor.Dispose(); owned.Clear(); Array.Clear(prefix); Array.Clear(ids); Array.Clear(text); Array.Clear(bytes); }
    private sealed record Bucket(int Frames, Tensor Memory, Tensor[] ProjectInputs, Tensor[] Cross, Tensor Mask, Tensor[] StepInputs);
}
