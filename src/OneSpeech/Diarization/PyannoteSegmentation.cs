using OneSpeech.Catalog;

namespace OneSpeech.Diarization;

/// <summary>Entry point for publisher-gated pyannote speech segmentation. Local authorized snapshots can supply union-of-speech intervals to ClusteringDiarizer, or local speaker activity to PyannoteDiarizer for overlap-aware turns.</summary>
/// <remarks>The network entry point still refuses missing credentials before I/O. Local loading verifies the pinned checkpoint without credentials or network access and writes converted graphs only in the user's cache. ClusteringDiarizer alone does not infer overlap from union-of-speech intervals.</remarks>
public static class PyannoteSegmentation
{
    /// <summary>Loads an authorized local segmentation-3.0 snapshot, verifying the publisher SHA-256 and converting locally into the specified cache (default ModelStore.Default.CacheDirectory). No network access, token, or writes to the snapshot. Threads default to 1. Throws ModelIntegrityException, I/O errors, ArgumentOutOfRangeException for nonpositive threads, or OperationCanceledException. Caller owns the returned detector.</summary>
    public static PyannoteSegmenter LoadLocal(string snapshotDirectory, string? cacheDirectory = null, int threads = 1, CancellationToken cancellationToken = default)
        => PyannoteSegmenter.LoadLocal(snapshotDirectory, cacheDirectory, threads, cancellationToken);

    /// <summary>Requests the segmentation slot by explicit model ID. Token precedence is argument, then ONESPEECH_MODEL_TOKEN. No token is retained. Always throws until network support is installed; no weights are fetched.</summary>
    /// <param name="model">One of the two catalogued pyannote IDs; defaults to segmentation-3.0, never a pipeline default.</param>
    /// <param name="token">Ephemeral publisher credential, or null to read the environment.</param>
    /// <param name="cancellationToken">Checked before validation.</param>
    /// <exception cref="ModelNotFoundException">The ID is not a supported segmentation entry.</exception>
    /// <exception cref="ModelTokenRequiredException">No nonempty credential is available.</exception>
    /// <exception cref="FeatureUnavailableException">Remote loading is unavailable. Use LoadLocal for an authorized segmentation-3.0 snapshot; use CommunityDiarizer.LoadLocal for an authorized community-1 snapshot.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static VoiceActivityDetector Load(string model = "pyannote/segmentation-3.0", string? token = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (model is not ("pyannote/segmentation-3.0" or "pyannote/speaker-diarization-community-1"))
            throw new ModelNotFoundException("Select pyannote/segmentation-3.0 or pyannote/speaker-diarization-community-1.");
        _ = ModelCatalog.Default.Get(model);
        token ??= Environment.GetEnvironmentVariable("ONESPEECH_MODEL_TOKEN");
        if (string.IsNullOrWhiteSpace(token)) throw new ModelTokenRequiredException("Accept the publisher gate and pass a token or set ONESPEECH_MODEL_TOKEN.");
        throw new FeatureUnavailableException("Remote pyannote segmentation loading is unavailable. Use PyannoteSegmentation.LoadLocal for an authorized segmentation-3.0 snapshot. Authorized community-1 files can be prefetched explicitly with ModelStore.PullAsync and loaded with CommunityDiarizer.LoadLocal.");
    }
}
