using System.Collections.Immutable;
using System.Security.Cryptography;
using OneSpeech.Catalog;
using RecursiveAI.OneModel;
using RecursiveAI.OneModel.Graph;
using RecursiveAI.OneModel.Tensors;

namespace OneSpeech.Diarization;

/// <summary>Publisher-gated segmentation-3.0 inference on CPU. The seven powerset classes describe up to three local speakers and two simultaneous speakers, not persistent speaker identities. Instances serialize inference and own their graph session.</summary>
public sealed class PyannoteSegmenter : VoiceActivityDetector
{
    private const string LicenseNotice = """
        MIT License

        Copyright (c) 2023 CNRS

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """;
    private readonly Session graph;
    private readonly object gate = new();
    private bool disposed;
    internal const int Samples = 160000;
    internal const int Frames = 589;
    internal static readonly int[] ClassMasks = [0, 1, 2, 4, 3, 5, 6];

    private PyannoteSegmenter(string path, ModelInfo model, int threads, CancellationToken cancellationToken)
    {
        Model = model;
        graph = Session.Load(path, new ModelOptions { IntraOpThreads = threads }, cancellationToken: cancellationToken);
    }

    /// <inheritdoc />
    public override ModelInfo Model { get; }

    /// <summary>Loads the pinned publisher snapshot without network access or credentials. Verifies the original checkpoint SHA-256, then converts it locally in cacheDirectory (default ModelStore.Default.CacheDirectory). The snapshot is read-only; neither upstream nor converted weights are redistributed. Threads default to 1. Throws ModelIntegrityException for a wrong checkpoint, I/O errors for missing files or concurrent conversion, ArgumentOutOfRangeException for nonpositive threads, and OperationCanceledException.</summary>
    public static PyannoteSegmenter LoadLocal(string snapshotDirectory, string? cacheDirectory = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDirectory);
        if (cacheDirectory is not null) ArgumentException.ThrowIfNullOrWhiteSpace(cacheDirectory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
        cancellationToken.ThrowIfCancellationRequested();
        return LoadCheckpoint(Path.Combine(snapshotDirectory, "pytorch_model.bin"), snapshotDirectory, "pyannote/segmentation-3.0", "pyannote-segmentation-3.0", LicenseNotice, cacheDirectory, threads, cancellationToken);
    }

    /// <summary>Verifies and converts one pinned publisher segmentation checkpoint; entry file 0 is the segmentation checkpoint. The notice is appended to the attribution when not null.</summary>
    internal static PyannoteSegmenter LoadCheckpoint(string upstream, string snapshotDirectory, string entryId, string cacheName, string? notice, string? cacheDirectory, int threads, CancellationToken cancellationToken)
    {
        var entry = ModelCatalog.Default.Get(entryId);
        using var file = File.OpenRead(upstream);
        if (file.Length != entry.Files[0].Size) throw new ModelIntegrityException("The local segmentation checkpoint has the wrong publisher file size.");
        string hash = Convert.ToHexStringLower(SHA256.HashData(file));
        if (hash != entry.Files[0].Sha256) throw new ModelIntegrityException("The local segmentation checkpoint does not match the pinned publisher SHA-256.");
        cancellationToken.ThrowIfCancellationRequested();
        string directory = Path.Combine(cacheDirectory ?? ModelStore.Default.CacheDirectory, cacheName, entry.Revision, "v1");
        string snapshot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(snapshotDirectory));
        string cache = Path.GetFullPath(directory);
        if (cache == snapshot || cache.StartsWith(snapshot + Path.DirectorySeparatorChar, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            throw new ArgumentException("The conversion cache must be outside the read-only publisher snapshot.", nameof(cacheDirectory));
        Directory.CreateDirectory(directory);
        using var lease = new FileStream(Path.Combine(directory, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        string converted = Path.Combine(directory, "segmentation.onemodel");
        // Rebuild from verified weights rather than trusting an unpinned pre-existing graph.
        file.Position = 0;
        string temporary = converted + ".partial";
        try
        {
            new PyannoteGraphConverter(TensorCheckpoint.Read(file, segmentation: true)).Save(temporary);
            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary, converted, overwrite: true);
            if (notice is not null) File.WriteAllText(Path.Combine(directory, "LICENSE.txt"), notice);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
        using var convertedFile = File.OpenRead(converted);
        string convertedHash = Convert.ToHexStringLower(SHA256.HashData(convertedFile));
        var model = new ModelInfo
        {
            Id = entry.Id, Source = entry.Source, Revision = entry.Revision, Tier = entry.Tier,
            License = entry.License, Attribution = entry.Attribution + "\nConverted locally to OneModel format." + (notice is null ? "" : "\n" + notice), Gated = entry.Gated,
            Files = [new ModelFileInfo { Name = entry.Files[0].Name, Sha256 = hash }, new ModelFileInfo { Name = "segmentation.onemodel", Sha256 = convertedHash }]
        };
        return new PyannoteSegmenter(converted, model, threads, cancellationToken);
    }

    /// <summary>Returns an owned row-major array of 589 frames × 7 log-probabilities for exactly 10 seconds of mono 16 kHz PCM. Class order: silence, speaker 1, speaker 2, speaker 3, pairs 1+2, 1+3, 2+3. Frame centers are (495 + 270 × frame)/16000 seconds. Throws ArgumentException for other audio shapes or nonfinite samples, ObjectDisposedException after disposal, and OperationCanceledException before/after inference.</summary>
    public float[] GetFrameLogProbabilities(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            if (audio.SampleRate != 16000 || audio.Channels != 1 || audio.Samples.Length != Samples)
                throw new ArgumentException("Segmentation requires exactly 160000 mono samples at 16000 Hz.", nameof(audio));
            foreach (float sample in audio.Samples.Span)
                if (!float.IsFinite(sample)) throw new ArgumentException("Audio must be finite.", nameof(audio));
            using var input = Tensor.Zeros(DType.F32, Shape.Create(1, 1, Samples));
            audio.Samples.Span.CopyTo(input.MutableData<float>());
            using var output = graph.Run(new Dictionary<string, Tensor> { ["waveform"] = input });
            cancellationToken.ThrowIfCancellationRequested();
            var result = output["log_probabilities"].Data<float>();
            if (result.Length != Frames * 7) throw new InvalidDataException("Segmentation graph returned an unexpected frame shape.");
            foreach (float value in result) if (!float.IsFinite(value)) throw new InvalidDataException("Segmentation returned nonfinite activations.");
            return result.ToArray();
        }
    }

    internal int[] Activity(AudioBuffer audio, CancellationToken token)
    {
        float[] probabilities = GetFrameLogProbabilities(audio, token);
        var activity = new int[Frames];
        for (int frame = 0; frame < Frames; frame++)
        {
            int best = 0;
            for (int c = 1; c < 7; c++) if (probabilities[frame * 7 + c] > probabilities[frame * 7 + best]) best = c;
            activity[frame] = ClassMasks[best];
        }
        return activity;
    }

    internal static int Boundary(int frame, int length) => frame == 0 ? 0 : frame >= Frames ? length : Math.Clamp(360 + frame * 270, 0, length);

    /// <summary>Returns the union of predicted speech, not speaker identities, using independent 10 s windows. Mono 16 kHz is required. A short final window is zero-padded; intervals are clipped to real audio. Empty audio returns no intervals. Cancellation is checked per window; invalid input throws ArgumentException.</summary>
    public override ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (audio.SampleRate != 16000 || audio.Channels != 1) throw new ArgumentException("Segmentation requires mono 16000 Hz PCM.", nameof(audio));
        cancellationToken.ThrowIfCancellationRequested();
        var intervals = new List<SpeechInterval>();
        int start = -1;
        for (int offset = 0; offset < audio.Samples.Length; offset += Samples)
        {
            int length = Math.Min(Samples, audio.Samples.Length - offset);
            var samples = new float[Samples]; audio.Samples.Span.Slice(offset, length).CopyTo(samples);
            int[] activity = Activity(new AudioBuffer(samples, 16000), cancellationToken);
            for (int frame = 0; frame < Frames; frame++)
            {
                int left = offset + Boundary(frame, length), right = offset + Boundary(frame + 1, length);
                if (right <= left) break;
                if (activity[frame] != 0) { if (start < 0) start = left; }
                else if (start >= 0) { intervals.Add(new SpeechInterval { Start = TimeSpan.FromSeconds(start / 16000.0), End = TimeSpan.FromSeconds(left / 16000.0) }); start = -1; }
            }
        }
        if (start >= 0) intervals.Add(new SpeechInterval { Start = TimeSpan.FromSeconds(start / 16000.0), End = audio.Duration });
        return [.. intervals];
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (!disposing) return;
        lock (gate) { if (disposed) return; disposed = true; graph.Dispose(); }
    }
}
