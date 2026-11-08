using System.Buffers.Binary;
using System.Collections.Immutable;
using OneSpeech.Catalog;

namespace OneSpeech.Diarization;

/// <summary>Overlap-aware, arrival-ordered Nemotron 3 Diarization for files or in-memory 16 kHz PCM, up to eight speakers.
/// Tier 2: explicitly selected, never a default; publisher weights and local-only runtime graphs remain in the user's cache.
/// Uses 27.2 s chunks plus 3.2 s right context. Working audio/model state is bounded; returned turn storage grows with speech changes.
/// Instances are not concurrent; dispose after processing. No speaker identities or embeddings are persisted.</summary>
public sealed class NemotronDiarizer : BlockDiarizer, IDisposable
{
    private readonly NemotronDiarizationSession session;
    private readonly float[] silence;
    private bool disposed;
    private int processing;
    /// <summary>Publisher and OpenMDW-1.1 agreement/origin provenance for this checkpoint.</summary>
    public ModelInfo Model { get; }
    /// <summary>Eight anonymous speaker channels; no live-session support is advertised by this file implementation.</summary>
    public override Capabilities Capabilities { get; } = new() { MaxSpeakers = 8 };
    /// <summary>The immutable provenance contributed to a transcription pipeline.</summary>
    public override ImmutableArray<ModelInfo> Models => [Model];

    private NemotronDiarizer(string directory, ModelInfo model, int threads, CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        NemotronDiarizationAssets.VerifyRuntime(directory, cancellationToken);
        byte[] bytes = File.ReadAllBytes(Path.Combine(directory, "silence.f32"));
        silence = new float[512];
        for (int i = 0; i < silence.Length; i++) silence[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        session = new NemotronDiarizationSession(directory, threads, cancellationToken);
        Model = model;
    }

    /// <summary>Pulls the named pinned publisher checkpoint through the supplied store (default ModelStore.Default), converts locally if needed,
    /// and loads CPU graphs. Never uses a mirror. threads defaults to zero (OneModel's default); negative values throw.
    /// Throws ModelIntegrityException on pin/digest failure, model/download errors on I/O failure, and OperationCanceledException.</summary>
    public static async ValueTask<NemotronDiarizer> LoadAsync(ModelStore? store = null, int threads = 0, CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        var resolved = await (store ?? ModelStore.Default).PullAsync(NemotronDiarizationAssets.Id, cancellationToken: cancellationToken).ConfigureAwait(false);
        return Load(resolved, threads, cancellationToken);
    }

    /// <summary>Loads an already resolved checkpoint from ModelStore. Verifies its exact publisher revision, SHA-256, tier and local runtime digests.
    /// threads defaults to zero (OneModel's default). Throws ModelIntegrityException on mismatch, ArgumentOutOfRangeException for negative threads,
    /// IOException for missing files, or OperationCanceledException. Does not download or convert.</summary>
    public static NemotronDiarizer Load(ResolvedModel model, int threads = 0, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        if (model.Info.Id != NemotronDiarizationAssets.Id || model.Info.Revision != NemotronDiarizationAssets.Revision || model.Info.Tier != ModelTier.Tier2)
            throw new ModelIntegrityException("Expected the pinned tier-2 nvidia/Nemotron-3-Diarization checkpoint.");
        NemotronDiarizationAssets.VerifyPublisher(model.Directory, cancellationToken);
        // Provenance is fixed by the publisher pin, not caller-editable metadata on ResolvedModel.
        return new NemotronDiarizer(model.Directory, new ModelInfo
        {
            Id = NemotronDiarizationAssets.Id, Source = "https://huggingface.co/nvidia/Nemotron-3-Diarization", Revision = NemotronDiarizationAssets.Revision,
            Tier = ModelTier.Tier2, License = "OpenMDW-1.1", Attribution = NemotronDiarizationAssets.Attribution,
            Files = [new ModelFileInfo { Name = "model.safetensors", Sha256 = NemotronDiarizationAssets.WeightSha },
                .. NemotronDiarizationAssets.RuntimeFiles.Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha })]
        }, threads, cancellationToken);
    }

    /// <summary>Returns overlapping turns for consecutive 16 kHz blocks, averaging channels. Empty input yields no turns.
    /// Each speaker is active at probability ≥0.5 on a 10 ms grid; endpoints are clipped to actual audio duration. Inputs shorter than one complete 10 ms hop return no turns.
    /// Throws UnsupportedAudioException on another rate, ArgumentException for nonfinite PCM/null blocks, InvalidOperationException on concurrent use,
    /// ObjectDisposedException after disposal, and OperationCanceledException at chunk boundaries.</summary>
    public override ImmutableArray<Turn> DiarizeBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken) =>
        NemotronActivitySegments.Coalesce(DiarizeActiveBlocks(blocks, cancellationToken));

    /// <summary>Like Diarize but retains per-interval activity ordering for TranscriptDiarization.Merge. Highest-probability active speaker gets relative activity 1,
    /// others 0.5; these are ranking values, not calibrated probabilities. Ties choose the lower channel index. Same limits/exceptions as DiarizeBlocks.</summary>
    public ImmutableArray<ActiveTurn> DiarizeActive(AudioBuffer audio, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        return DiarizeActiveBlocks([audio], cancellationToken);
    }

    /// <summary>Block form of DiarizeActive. Keeps only bounded waveform/features/cache state plus the returned change-point intervals.
    /// Same limits, ranking semantics, exceptions and cancellation as DiarizeBlocks. No live events are emitted.</summary>
    public ImmutableArray<ActiveTurn> DiarizeActiveBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (Interlocked.Exchange(ref processing, 1) != 0) throw new InvalidOperationException("Use separate Nemotron diarizers for concurrent recordings.");
        try
        {
            var cache = new ArrivalOrderSpeakerCache(silence);
            var segments = new NemotronActivitySegments();
            long firstMelFrame = 0;
            foreach (var chunk in NemotronFeatureChunks.Read(blocks, cancellationToken))
            {
                if (chunk.ValidFrames == 0) continue; // No complete 10 ms frame exists in this final sub-hop tail.
                int prefix = cache.CachedFrames + cache.FifoFrames;
                float[] window = cache.Window(session.Embed(chunk.Stacked, cancellationToken));
                var result = session.Forward(window, cancellationToken, validFrames: prefix + chunk.ValidFrames);
                segments.Append(result.Probabilities.AsSpan(prefix * 64), chunk.OutputMelFrames, firstMelFrame, chunk.SampleEnd);
                cache.Update(window, result.Probabilities, Math.Min(340, chunk.Frames));
                firstMelFrame += chunk.OutputMelFrames;
            }
            return segments.Complete();
        }
        finally { Volatile.Write(ref processing, 0); }
    }

    /// <summary>Releases CPU graph state. Idempotent; must not run concurrently with processing.</summary>
    public void Dispose()
    {
        if (disposed) return;
        disposed = true; session.Dispose();
    }
}
