using System.Collections.Immutable;
using System.Security.Cryptography;
using OneSpeech.Catalog;

namespace OneSpeech.Diarization;

/// <summary>Overlap-aware offline diarization with the published community-1 models on CPU: powerset segmentation over
/// 10 s windows every second, WeSpeaker ResNet34 embeddings of each local speaker's overlap-free speech, the published
/// x-vector transform and PLDA, VBx clustering, and reconstruction by speaker count. Tier 2 (CC-BY-4.0, gated): loaded
/// only from an authorized local copy of the publisher's snapshot at the pinned revision, never a default, never
/// redistributed. Embeddings are never persisted. Diarization can disagree with the reference pipeline where its
/// unpublished choices differ; docs/diarization.md lists the measured agreement.</summary>
/// <remarks>The whole recording is held in memory as mono 16 kHz samples, and clustering keeps one 256-element embedding
/// per active local speaker and window plus an O(n²) distance matrix over them; budget accordingly for multi-hour audio.</remarks>
public sealed class CommunityDiarizer : BlockDiarizer
{
    private const string EntryId = "pyannote/speaker-diarization-community-1";
    private readonly PyannoteSegmenter segmenter;
    private readonly WeSpeakerEmbedder embedder;
    private readonly CommunityPipeline pipeline;
    private readonly ModelInfo model;
    private readonly object gate = new();

    private CommunityDiarizer(PyannoteSegmenter segmenter, WeSpeakerEmbedder embedder, PldaTransform plda, ModelInfo model)
    {
        this.segmenter = segmenter; this.embedder = embedder; this.model = model;
        pipeline = new CommunityPipeline(segmenter, embedder, plda, new CommunityOptions());
    }

    /// <summary>Loads an authorized local copy of the pinned community-1 snapshot (the directory holding
    /// <c>segmentation/pytorch_model.bin</c>, <c>embedding/pytorch_model.bin</c>, <c>plda/plda.npz</c> and
    /// <c>plda/xvec_transform.npz</c>) without network access or credentials. Every file's size and SHA-256 is checked
    /// against the catalog pin, then the networks are converted into <paramref name="cacheDirectory"/> (default: the
    /// model cache). The snapshot is read-only and nothing from it is copied or redistributed. Threads default to 1.
    /// Throws <see cref="ModelIntegrityException"/> for a wrong file, I/O errors for missing files,
    /// <see cref="InvalidDataException"/> for malformed content, ArgumentOutOfRangeException for nonpositive threads and
    /// OperationCanceledException. The caller owns the result.</summary>
    public static CommunityDiarizer LoadLocal(string snapshotDirectory, string? cacheDirectory = null, int threads = 1, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(snapshotDirectory);
        if (cacheDirectory is not null) ArgumentException.ThrowIfNullOrWhiteSpace(cacheDirectory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
        cancellationToken.ThrowIfCancellationRequested();
        var entry = ModelCatalog.Default.Get(EntryId);
        var verified = new List<ModelFileInfo>();
        foreach (var file in entry.Files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string path = Path.Combine(snapshotDirectory, file.Name);
            using var stream = File.OpenRead(path);
            if (stream.Length != file.Size) throw new ModelIntegrityException("The local community-1 file " + file.Name + " has the wrong publisher file size.");
            string hash = Convert.ToHexStringLower(SHA256.HashData(stream));
            if (hash != file.Sha256) throw new ModelIntegrityException("The local community-1 file " + file.Name + " does not match the pinned publisher SHA-256.");
            verified.Add(new ModelFileInfo { Name = file.Name, Sha256 = hash });
        }
        string cacheRoot = cacheDirectory ?? ModelStore.Default.CacheDirectory;
        string snapshot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(snapshotDirectory));
        string directory = Path.GetFullPath(Path.Combine(cacheRoot, "pyannote-community-1", entry.Revision, "v1"));
        if (directory == snapshot || directory.StartsWith(snapshot + Path.DirectorySeparatorChar, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            throw new ArgumentException("The conversion cache must be outside the read-only publisher snapshot.", nameof(cacheDirectory));
        var segmenter = PyannoteSegmenter.LoadCheckpoint(Path.Combine(snapshotDirectory, entry.Files[0].Name), snapshotDirectory, EntryId, "pyannote-community-1", null, cacheRoot, threads, cancellationToken);
        try
        {
            string graph = Path.Combine(directory, "wespeaker-frames.onemodel");
            WeSpeakerResNetConverter converter;
            using (var input = File.OpenRead(Path.Combine(snapshotDirectory, entry.Files[1].Name)))
                converter = new WeSpeakerResNetConverter(TensorCheckpoint.Read(input, segmentation: true));
            string partial = graph + ".partial";
            try { converter.Save(partial); File.Move(partial, graph, overwrite: true); }
            finally { if (File.Exists(partial)) File.Delete(partial); }
            var (projection, bias) = converter.Projection();
            var plda = PldaTransform.Load(Path.Combine(snapshotDirectory, entry.Files[2].Name), Path.Combine(snapshotDirectory, entry.Files[3].Name));
            var embedder = new WeSpeakerEmbedder(graph, projection, bias, threads, cancellationToken);
            using var graphFile = File.OpenRead(graph);
            verified.Add(new ModelFileInfo { Name = "wespeaker-frames.onemodel", Sha256 = Convert.ToHexStringLower(SHA256.HashData(graphFile)) });
            foreach (var f in segmenter.Model.Files) if (f.Name.EndsWith(".onemodel", StringComparison.Ordinal)) verified.Add(f);
            var info = new ModelInfo
            {
                Id = entry.Id, Source = entry.Source, Revision = entry.Revision, Tier = entry.Tier, License = entry.License, Gated = entry.Gated,
                Attribution = entry.Attribution + "\nConverted locally to OneModel format. Components: pyannote segmentation (Plaquet and Bredin), WeSpeaker ResNet34-LM embedding (Wang et al.), x-vector transform and PLDA by BUT Speech@FIT (Landini et al.).",
                Files = [.. verified],
            };
            return new CommunityDiarizer(segmenter, embedder, plda, info);
        }
        catch { segmenter.Dispose(); throw; }
    }

    /// <summary>No fixed speaker limit; the clustering decides the number of speakers. At most two simultaneous speakers per frame survive reconstruction only if the segmentation model reports them, up to three local speakers per 10 s window.</summary>
    public override Capabilities Capabilities => new();

    /// <inheritdoc />
    public override ImmutableArray<ModelInfo> Models => [model];

    /// <summary>Diarizes consecutive 16 kHz blocks of up to 32 channels (averaged to mono) as one recording and returns
    /// turns that may overlap, on the input clock, with speakers numbered by first appearance in the clustering.
    /// Holds the whole recording in memory. Audio shorter than 25 ms yields no turns. Throws ArgumentException for other
    /// sample rates or changing channel counts, ObjectDisposedException after disposal, and OperationCanceledException.</summary>
    public override ImmutableArray<Turn> DiarizeBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        var samples = new List<float>();
        foreach (var block in blocks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (block.SampleRate != 16000 || block.Channels < 1 || block.Channels > 32) throw new ArgumentException("Diarization requires 16 kHz PCM with 1–32 channels.", nameof(blocks));
            int channels = block.Channels, frames = block.Samples.Length / channels;
            var span = block.Samples.Span;
            for (int i = 0; i < frames; i++)
            {
                double value = 0;
                for (int c = 0; c < channels; c++) value += span[i * channels + c];
                float v = (float)(value / channels);
                if (!float.IsFinite(v)) throw new ArgumentException("Audio must be finite.", nameof(blocks));
                samples.Add(v);
            }
        }
        if (samples.Count < 400) return [];
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return pipeline.Run([.. samples], cancellationToken).Turns;
        }
    }

    private bool disposed;

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (!disposing) return;
        lock (gate) { if (disposed) return; disposed = true; segmenter.Dispose(); embedder.Dispose(); }
    }
}
