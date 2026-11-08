using OneSpeech.Catalog;

namespace OneSpeech.Diarization;

/// <summary>Entry point for publisher-gated pyannote speech segmentation. The clustering slot is the existing VoiceActivityDetector argument of ClusteringDiarizer; a segmentation implementation supplies its union-of-speech intervals through that role.</summary>
/// <remarks>No network implementation is provided yet. This entry point refuses missing credentials before any I/O and never downloads unsupported weights. Supplying credentials does not assert gate acceptance; only a publisher response can do that. The clustering engine does not infer overlap from union-of-speech intervals.</remarks>
public static class PyannoteSegmentation
{
    /// <summary>Requests the segmentation slot by explicit model ID. Token precedence is argument, then ONESPEECH_MODEL_TOKEN. No token is retained. Always throws until network support is installed; no weights are fetched.</summary>
    /// <param name="model">One of the two catalogued pyannote IDs; defaults to segmentation-3.0, never a pipeline default.</param>
    /// <param name="token">Ephemeral publisher credential, or null to read the environment.</param>
    /// <param name="cancellationToken">Checked before validation.</param>
    /// <exception cref="ModelNotFoundException">The ID is not a supported segmentation entry.</exception>
    /// <exception cref="ModelTokenRequiredException">No nonempty credential is available.</exception>
    /// <exception cref="FeatureUnavailableException">The segmentation network is not implemented. Use a VoiceActivityDetector implementation in ClusteringDiarizer in the meantime.</exception>
    /// <exception cref="OperationCanceledException">Cancellation was requested.</exception>
    public static VoiceActivityDetector Load(string model = "pyannote/segmentation-3.0", string? token = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (model is not ("pyannote/segmentation-3.0" or "pyannote/speaker-diarization-community-1"))
            throw new ModelNotFoundException("Select pyannote/segmentation-3.0 or pyannote/speaker-diarization-community-1.");
        _ = ModelCatalog.Default.Get(model);
        token ??= Environment.GetEnvironmentVariable("ONESPEECH_MODEL_TOKEN");
        if (string.IsNullOrWhiteSpace(token)) throw new ModelTokenRequiredException("Accept the publisher gate and pass a token or set ONESPEECH_MODEL_TOKEN.");
        throw new FeatureUnavailableException("The pyannote segmentation network is not implemented. Supply a speech-interval detector to ClusteringDiarizer; model files can be prefetched explicitly with ModelStore.PullAsync.");
    }
}
