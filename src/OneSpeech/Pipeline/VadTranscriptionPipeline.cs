using System.Collections.Immutable;
using OneSpeech.Audio;
using OneSpeech.Diarization;

namespace OneSpeech.Pipeline;

/// <summary>Combines a VAD and recognizer on one audio clock. Chunk text is joined by one space; no timings are invented. Models are borrowed and must outlive processing.</summary>
public sealed class VadTranscriptionPipeline : TranscriptionPipeline
{
    private readonly SpeechRecognizer recognizer;
    private readonly VoiceActivityDetector detector;
    private readonly VadChunker chunker;
    private readonly IProgress<TimeSpan>? progress;
    private readonly Diarizer? diarizer;

    /// <summary>Creates a pipeline. Maximum chunks default to 30 s (1–3600 s); progress includes silence in processed audio time. Recognition averages input channels; original file channels remain available through AudioFileSource.</summary>
    public VadTranscriptionPipeline(SpeechRecognizer recognizer, VoiceActivityDetector detector, TimeSpan? maximumChunkDuration = null, IProgress<TimeSpan>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(recognizer);
        ArgumentNullException.ThrowIfNull(detector);
        this.recognizer = recognizer;
        this.detector = detector;
        this.progress = progress;
        chunker = new VadChunker(detector, maximumChunkDuration);
    }

    /// <summary>Creates a pipeline with a whole-recording diarization pass. Models are borrowed.
    /// File/encoded-stream inputs require a BlockDiarizer and make two bounded decoding passes; no chunk-local speaker IDs are substituted.</summary>
    public VadTranscriptionPipeline(SpeechRecognizer recognizer, VoiceActivityDetector detector, TimeSpan? maximumChunkDuration, IProgress<TimeSpan>? progress, Diarizer diarizer)
        : this(recognizer, detector, maximumChunkDuration, progress)
    {
        ArgumentNullException.ThrowIfNull(diarizer);
        this.diarizer = diarizer;
    }

    /// <summary>Transcribes 16 kHz interleaved PCM. Other rates throw UnsupportedAudioException; empty input returns an empty document. Cancellation is checked at every chunk.</summary>
    public override Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var transcript = Combine([audio], language, cancellationToken);
        return MergeDocument(transcript, diarizer?.Diarize(audio, cancellationToken).Select(t => new ActiveTurn { Turn = t }).ToArray(), cancellationToken);
    }

    /// <summary>Decodes and transcribes a file without retaining its PCM. The returned document necessarily grows with the recognized text. Filesystem errors remain IOException; corrupt/unsupported audio throws UnsupportedAudioException.</summary>
    public Transcript Transcribe(string path, string? language, CancellationToken cancellationToken)
    {
        var turns = FileTurns(path, cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        return MergeDocument(Combine(VadChunker.Blocks(source, cancellationToken), language, cancellationToken), turns, cancellationToken);
    }

    /// <summary>Asynchronously copies caller-owned encoded input into a temporary seekable file, then decodes bounded chunks synchronously. The spool is deleted on completion/cancellation. No background inference or synchronous input-stream reads are used.</summary>
    public async ValueTask<Transcript> TranscribeAsync(Stream stream, string? language, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var blockDiarizer = RequireBlockDiarizer();
        if (!stream.CanRead) throw new ArgumentException("Input stream must be readable.", nameof(stream));
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        await using var spool = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 65536, FileOptions.Asynchronous | FileOptions.DeleteOnClose);
        await stream.CopyToAsync(spool, cancellationToken).ConfigureAwait(false);
        spool.Position = 0;
        ActiveTurn[]? turns = null;
        if (blockDiarizer is not null)
        {
            using var firstPass = AudioFileSource.Open(spool, true, cancellationToken);
            turns = blockDiarizer.DiarizeBlocks(VadChunker.Blocks(firstPass, cancellationToken), cancellationToken).Select(t => new ActiveTurn { Turn = t }).ToArray();
        }
        spool.Position = 0;
        using var source = AudioFileSource.Open(spool, true, cancellationToken);
        return MergeDocument(Combine(VadChunker.Blocks(source, cancellationToken), language, cancellationToken), turns, cancellationToken);
    }

    /// <summary>Yields final segments on the input clock from bounded 16 kHz chunks, without accumulating the document. Cancellation is checked per chunk; stereo is averaged.</summary>
    public override IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var activeTurns = diarizer?.Diarize(audio, cancellationToken).Select(t => new ActiveTurn { Turn = t }).ToArray();
        foreach (var chunk in chunker.Read([audio], progress, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var segment in recognizer.TranscribeStream(chunk.Audio, language, cancellationToken))
            {
                language ??= segment.Language;
                var shifted = Shift(segment, chunk.Start);
                if (activeTurns is not null)
                {
                    var merged = TranscriptDiarization.Merge(new Transcript { Text = shifted.Text, Segments = [shifted] }, activeTurns, cancellationToken);
                    shifted = merged.Segments[0];
                }
                yield return shifted;
            }
        }
    }

    /// <summary>Decodes a file and yields final segments without retaining its PCM or accumulating text. Enumeration owns and closes the decoder, including on early disposal. Errors and cancellation follow AudioFileSource.</summary>
    public IEnumerable<Segment> TranscribeStream(string path, string? language, CancellationToken cancellationToken)
    {
        var activeTurns = FileTurns(path, cancellationToken);
        using var source = AudioFileSource.Open(path, cancellationToken);
        foreach (var chunk in chunker.Read(VadChunker.Blocks(source, cancellationToken), progress, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            foreach (var segment in recognizer.TranscribeStream(chunk.Audio, language, cancellationToken))
            {
                language ??= segment.Language;
                var shifted = Shift(segment, chunk.Start);
                if (activeTurns is not null)
                    shifted = TranscriptDiarization.Merge(new Transcript { Text = shifted.Text, Segments = [shifted] }, activeTurns, cancellationToken).Segments[0];
                yield return shifted;
            }
        }
    }

    private BlockDiarizer? RequireBlockDiarizer()
    {
        if (diarizer is null) return null;
        return diarizer as BlockDiarizer ?? throw new CapabilityNotSupportedException("File diarization requires a BlockDiarizer, such as ClusteringDiarizer or ChannelDiarizer. Supply a block-capable implementation or use in-memory audio.");
    }

    private ActiveTurn[]? FileTurns(string path, CancellationToken cancellationToken)
    {
        var blockDiarizer = RequireBlockDiarizer();
        if (blockDiarizer is null) return null;
        using var source = AudioFileSource.Open(path, cancellationToken);
        return blockDiarizer.DiarizeBlocks(VadChunker.Blocks(source, cancellationToken), cancellationToken).Select(t => new ActiveTurn { Turn = t }).ToArray();
    }

    private Transcript MergeDocument(Transcript transcript, ActiveTurn[]? turns, CancellationToken cancellationToken)
    {
        if (turns is null) return transcript;
        var merged = TranscriptDiarization.Merge(transcript, turns, cancellationToken);
        return diarizer is BlockDiarizer block ? merged with { Models = [.. merged.Models.Concat(block.Models).DistinctBy(m => (m.Id, m.Revision))] } : merged;
    }

    internal Transcript TranscribeBlocks(IEnumerable<AudioBuffer> blocks, string? language, CancellationToken cancellationToken) => Combine(blocks, language, cancellationToken);

    private Transcript Combine(IEnumerable<AudioBuffer> blocks, string? language, CancellationToken cancellationToken)
    {
        var words = ImmutableArray.CreateBuilder<Word>();
        var segments = ImmutableArray.CreateBuilder<Segment>();
        var turns = ImmutableArray.CreateBuilder<Turn>();
        var models = new[] { detector.Model, recognizer.Model }.DistinctBy(m => (m.Id, m.Revision)).ToList();
        var text = new List<string>();
        TimeSpan duration = TimeSpan.Zero;
        var clock = new AudioProgress(value => { duration = value; progress?.Report(value); });
        foreach (var chunk in chunker.Read(blocks, clock, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var result = recognizer.Transcribe(chunk.Audio, language, cancellationToken);
            if (result.Text.Length != 0) text.Add(result.Text);
            language ??= result.Language;
            words.AddRange(result.Words.Select(w => Shift(w, chunk.Start)));
            segments.AddRange(result.Segments.Select(s => Shift(s, chunk.Start)));
            turns.AddRange(result.Turns.Select(t => t with { Start = t.Start + chunk.Start, End = t.End + chunk.Start, Words = [.. t.Words.Select(w => Shift(w, chunk.Start))] }));
            foreach (var model in result.Models)
                if (!models.Exists(m => m.Id == model.Id && m.Revision == model.Revision)) models.Add(model);
        }
        return new Transcript
        {
            Text = string.Join(" ", text), Language = language, Duration = duration,
            Words = words.ToImmutable(), Segments = segments.ToImmutable(), Turns = turns.ToImmutable(), Models = [.. models]
        };
    }

    private static Word Shift(Word word, TimeSpan offset) => word with { Start = word.Start + offset, End = word.End + offset };
    private static Segment Shift(Segment segment, TimeSpan offset) => segment with
    {
        Start = segment.Start + offset, End = segment.End + offset, Words = [.. segment.Words.Select(w => Shift(w, offset))]
    };
    private sealed class AudioProgress(Action<TimeSpan> report) : IProgress<TimeSpan>
    {
        public void Report(TimeSpan value) => report(value);
    }
}
