using System.Collections.Immutable;

namespace OneSpeech.Diarization;

/// <summary>Offline tier-1 voice clustering: VAD, speech-only overlapping embeddings, spectral clustering or cosine AHC.
/// Does not detect overlapped speech. Models are borrowed and must outlive processing; no embeddings are persisted.</summary>
public sealed class ClusteringDiarizer : BlockDiarizer
{
    private readonly SpeakerEmbedder embedder;
    private readonly VoiceActivityDetector detector;
    private readonly ClusteringOptions options;
    private readonly EmbeddingResegmenter? resegmenter;

    /// <summary>Creates a diarizer for 16 kHz PCM, averaged over input channels. Defaults are ClusteringOptions defaults and cosine resegmentation (penalty 0.4, radius 2).
    /// The supplied detector is never reconfigured; use SileroVadOptions.Diarization for the file-helper activity profile.
    /// Embedder must require 16 kHz with dimension 1–4096. Throws ArgumentException for invalid options and CapabilityNotSupportedException for unsupported embedder shape/rate.</summary>
    public ClusteringDiarizer(SpeakerEmbedder embedder, VoiceActivityDetector detector, ClusteringOptions? options = null, EmbeddingResegmenter? resegmenter = null)
    {
        ArgumentNullException.ThrowIfNull(embedder); ArgumentNullException.ThrowIfNull(detector);
        this.options = options ?? new ClusteringOptions(); this.options.Validate();
        if (embedder.SampleRate != 16000 || embedder.Dimension < 1 || embedder.Dimension > 4096)
            throw new CapabilityNotSupportedException("Clustering requires a 16 kHz speaker embedder with dimension 1–4096.");
        this.embedder = embedder; this.detector = detector;
        // Unconstrained smoothing can remove a label. Preserve caller count hints without
        // inventing a replacement label; explicit custom backends retain strict validation.
        this.resegmenter = resegmenter ?? (this.options.Resegment && this.options.NumSpeakers is null && this.options.MinSpeakers == 1
            ? new CosineResegmenter(0.4, 2) : null);
        Capabilities = new Capabilities { MaxSpeakers = this.options.MaxSpeakers };
    }

    /// <summary>Maximum configured speaker count; no live streaming or overlap-detection claim.</summary>
    public override Capabilities Capabilities { get; }
    /// <summary>Contributing VAD and embedding model provenance, deduplicated by ID/revision.</summary>
    public override ImmutableArray<ModelInfo> Models => [.. new[] { detector.Model, embedder.Model }.DistinctBy(m => (m.Id, m.Revision))];

    /// <summary>Processes consecutive 16 kHz blocks with consistent channel count (1–32), using 30 s independent VAD horizons.
    /// Retains O(ND) embedding storage for N windows of dimension D plus O(L²) dense landmark work (L ≤256); PCM is bounded to two 30 s interleaved buffers plus one mono buffer.
    /// At most MaxWindows embeddings are retained; exceeding this throws CapabilityNotSupportedException. Empty/no speech yields no turns. Cancels per block/window and clustering sweep.</summary>
    public override ImmutableArray<Turn> DiarizeBlocks(IEnumerable<AudioBuffer> blocks, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        var windows = new List<EmbeddingWindow>();
        var ownership = new List<(TimeSpan Start, TimeSpan End)>();
        long offsetFrames = 0;
        int windowLength = (int)Math.Round(options.WindowSeconds * 16000), shift = (int)Math.Round(options.ShiftSeconds * 16000);
        foreach (var block in DiarizationBlocks.Fixed(blocks, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            ReadOnlyMemory<float> mono = block.Samples;
            if (block.Channels != 1)
            {
                var mixed = new float[block.FrameCount];
                for (int i = 0; i < mixed.Length; i++) { double sum = 0; for (int channel = 0; channel < block.Channels; channel++) sum += block.Samples.Span[i * block.Channels + channel]; mixed[i] = (float)(sum / block.Channels); }
                mono = mixed;
            }
            var audio = new AudioBuffer(mono, 16000);
            var intervals = detector.Detect(audio, cancellationToken).OrderBy(i => i.Start).ToArray();
            foreach (var interval in intervals)
                if (interval.Start < TimeSpan.Zero || interval.End <= interval.Start || interval.End > audio.Duration)
                    throw new InvalidOperationException("VAD returned an interval outside its analysis horizon.");
            var regions = new List<(int Start, int End)>();
            foreach (var interval in intervals)
            {
                int start = (int)Math.Round(interval.Start.TotalSeconds * 16000), end = (int)Math.Round(interval.End.TotalSeconds * 16000);
                if (regions.Count != 0 && start <= regions[^1].End) regions[^1] = (regions[^1].Start, Math.Max(end, regions[^1].End));
                else regions.Add((start, end));
            }
            TimeSpan At(int frame) => TimeSpan.FromSeconds((double)(offsetFrames + frame) / 16000);
            foreach (var region in regions)
            {
                int first = windows.Count;
                // Landini et al. §3.2: 100 ms minimum; the internal diagnostic can ablate this cutoff.
                for (int start = region.Start; start < region.End; start += shift)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int length = Math.Min(windowLength, region.End - start);
                    if (length < (int)Math.Round(options.MinimumSegmentSeconds * 16000)) break;
                    if (windows.Count == options.MaxWindows) throw new CapabilityNotSupportedException("Diarization exceeded MaxWindows. Increase ClusteringOptions.MaxWindows within its documented bound or split the input explicitly.");
                    var vector = new float[embedder.Dimension];
                    embedder.Embed(new AudioBuffer(mono.Slice(start, length), 16000), vector, cancellationToken);
                    EmbeddingClustering.Normalize(vector);
                    windows.Add(new EmbeddingWindow { Start = At(start), End = At(start + length), Embedding = vector });
                }
                for (int i = first; i < windows.Count; i++)
                {
                    TimeSpan Center(int index) => windows[index].Start + (windows[index].End - windows[index].Start) / 2;
                    var start = i == first ? At(region.Start) : Center(i - 1) + (Center(i) - Center(i - 1)) / 2;
                    var end = i + 1 == windows.Count ? At(region.End) : Center(i) + (Center(i + 1) - Center(i)) / 2;
                    ownership.Add((start, end));
                }
            }
            offsetFrames += block.FrameCount;
        }
        var labels = EmbeddingClustering.Cluster(windows, options, cancellationToken);
        if (labels.Length == 0) return [];
        if (resegmenter is not null)
        {
            int count = labels.Max() + 1;
            var refined = resegmenter.Resegment(windows, labels, cancellationToken);
            if (refined is null || refined.Length != labels.Length || refined.Any(l => l < 0 || l >= count)) throw new InvalidOperationException("Resegmenter returned invalid labels.");
            int speakers = refined.Distinct().Count();
            if (speakers < options.MinSpeakers || speakers > options.MaxSpeakers || options.NumSpeakers is {} exact && speakers != exact)
                throw new InvalidOperationException("Resegmenter violated the requested speaker count.");
            labels = EmbeddingClustering.Canonical(refined);
        }
        var result = ImmutableArray.CreateBuilder<Turn>();
        for (int i = 0; i < labels.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var speaker = new SpeakerLabel { Id = "speaker-" + (labels[i] + 1).ToString(System.Globalization.CultureInfo.InvariantCulture) };
            if (result.Count != 0 && result[^1].Speaker.Id == speaker.Id && result[^1].End == ownership[i].Start) result[^1] = result[^1] with { End = ownership[i].End };
            else result.Add(new Turn { Speaker = speaker, Start = ownership[i].Start, End = ownership[i].End, Text = "" });
        }
        return result.ToImmutable();
    }
}

internal static class DiarizationBlocks
{
    internal static IEnumerable<AudioBuffer> Fixed(IEnumerable<AudioBuffer> blocks, CancellationToken token)
    {
        float[]? buffer = null; int channels = 0, used = 0;
        foreach (var block in blocks)
        {
            token.ThrowIfCancellationRequested();
            if (block.SampleRate != 16000 || block.Channels < 1 || block.Channels > 32 || channels != 0 && channels != block.Channels)
                throw new UnsupportedAudioException("Diarization requires consecutive 16 kHz PCM blocks with a constant channel count in 1–32.");
            channels = block.Channels; buffer ??= new float[30 * 16000 * channels];
            int offset = 0;
            while (offset < block.Samples.Length)
            {
                token.ThrowIfCancellationRequested();
                int length = Math.Min(buffer.Length - used, block.Samples.Length - offset);
                block.Samples.Slice(offset, length).CopyTo(buffer.AsMemory(used)); used += length; offset += length;
                if (used == buffer.Length) { yield return new AudioBuffer(buffer, 16000, channels); used = 0; }
            }
        }
        if (used != 0) yield return new AudioBuffer(buffer!.AsMemory(0, used), 16000, channels);
    }
}
