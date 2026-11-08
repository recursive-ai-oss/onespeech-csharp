using System.Collections.Immutable;

namespace OneSpeech;

/// <summary>A detected speech interval on the input audio clock.</summary>
public sealed record SpeechInterval
{
    /// <summary>Inclusive start relative to input audio.</summary>
    public required TimeSpan Start { get; init; }
    /// <summary>Exclusive end relative to input audio.</summary>
    public required TimeSpan End { get; init; }
}

/// <summary>Detects speech in memory. Implementations document supported rates, channels and cancellation granularity.</summary>
public abstract class VoiceActivityDetector : IDisposable
{
    /// <summary>Releases resources owned by this role. Borrowed collaborators are not disposed. Recognizer decorators, including phrase-bias wrappers, must never dispose their inner recognizer. Repeated calls are harmless.</summary>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    /// <summary>Releases owned managed resources when disposing is true. Wrappers must not dispose borrowed roles.</summary>
    protected virtual void Dispose(bool disposing) { }

    /// <summary>Model provenance for this detector.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Returns speech intervals; empty audio yields no intervals. Throws OperationCanceledException on cancellation.</summary>
    public abstract ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken = default);
}

/// <summary>Recognizes in-memory audio synchronously. Live sessions own independent per-input state.</summary>
public abstract class SpeechRecognizer : IDisposable
{
    /// <summary>Releases resources owned by this role. Borrowed collaborators are not disposed. Repeated calls are harmless.</summary>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    /// <summary>Releases owned managed resources when disposing is true. Wrappers must not dispose borrowed roles.</summary>
    protected virtual void Dispose(bool disposing) { }

    /// <summary>Creates a finals-only, VAD-endpointed live session by default. Stateful streaming families override this to supply true partials. Borrows this recognizer and the detector; dispose the session first. Input must be mono 16 kHz. Options and errors follow UtteranceLiveSession.</summary>
    public virtual LiveSession CreateLiveSession(VoiceActivityDetector detector, string? language = null) => new Live.UtteranceLiveSession(this, detector, language);
    /// <summary>Loaded model provenance.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Requests outside this declaration throw CapabilityNotSupportedException.</summary>
    public abstract Capabilities Capabilities { get; }
    /// <summary>Transcribes audio, with a language tag or null for model detection. Throws OperationCanceledException on cancellation.</summary>
    public abstract Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default);
    /// <summary>Yields final immutable segments in audio order, using bounded chunks. Throws OperationCanceledException on cancellation.</summary>
    public abstract IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default);
    /// <summary>Detects a language tag; throws CapabilityNotSupportedException if unsupported and OperationCanceledException on cancellation.</summary>
    public abstract string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken = default);
}

/// <summary>Aligns written text with in-memory audio without inventing failed timings.</summary>
public abstract class Aligner : IDisposable
{
    /// <summary>Releases resources owned by this role. Borrowed collaborators are not disposed. Repeated calls are harmless.</summary>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    /// <summary>Releases owned managed resources when disposing is true. Wrappers must not dispose borrowed roles.</summary>
    protected virtual void Dispose(bool disposing) { }

    /// <summary>Loaded model provenance.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Supported languages and timing features.</summary>
    public abstract Capabilities Capabilities { get; }
    /// <summary>Aligns text to audio. Null language requests detection. Throws OperationCanceledException on cancellation.</summary>
    public abstract Transcript Align(AudioBuffer audio, string text, string? language = null, CancellationToken cancellationToken = default);
    /// <summary>Aligns existing written words without changing their text, order or non-timing fields. The default delegates to text alignment and projects timing only when the word count is unchanged; otherwise every word remains untimed. Implementations can override this to retain explicit word boundaries through spoken-form expansion. Errors follow text alignment; cancellation throws OperationCanceledException.</summary>
    public virtual Transcript Align(AudioBuffer audio, Transcript transcript, string? language = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        cancellationToken.ThrowIfCancellationRequested();
        var timed = Align(audio, string.Join(' ', transcript.Words.Select(w => w.Text)), language, cancellationToken);
        var words = transcript.Words.Select((word, i) => timed.Words.Length == transcript.Words.Length
            ? word with { Start = timed.Words[i].Start, End = timed.Words[i].End, TimingMethod = timed.Words[i].TimingMethod }
            : word with { Start = null, End = null, TimingMethod = TimingMethod.None }).ToImmutableArray();
        return Alignment.AlignedWords.Replace(transcript, words) with { Duration = timed.Duration, Models = [.. transcript.Models.Concat(timed.Models).DistinctBy(m => (m.Id, m.Revision))] };
    }
}

/// <summary>Assigns speaker turns to in-memory audio; implementations declare whether overlap is detected.</summary>
public abstract class Diarizer : IDisposable
{
    /// <summary>Releases resources owned by this role. Borrowed collaborators are not disposed. Repeated calls are harmless.</summary>
    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    /// <summary>Releases owned managed resources when disposing is true. Wrappers must not dispose borrowed roles.</summary>
    protected virtual void Dispose(bool disposing) { }

    /// <summary>Supported speaker count and other limits.</summary>
    public abstract Capabilities Capabilities { get; }
    /// <summary>Returns turns on the input audio clock; throws OperationCanceledException on cancellation.</summary>
    public abstract ImmutableArray<Turn> Diarize(AudioBuffer audio, CancellationToken cancellationToken = default);
}

/// <summary>An identification or verification score; score scale is implementation-defined, not a probability.</summary>
public sealed record SpeakerMatch
{
    /// <summary>Matched enrolled ID, or null for unknown.</summary>
    public string? SpeakerId { get; init; }
    /// <summary>Similarity score on the registry's documented scale.</summary>
    public required double Score { get; init; }
}

/// <summary>In-memory speaker enrollment only. Implementations never persist embeddings or enrollment audio.</summary>
public abstract class SpeakerRegistry
{
    /// <summary>Enrolls or replaces a named speaker; throws OperationCanceledException on cancellation.</summary>
    public abstract void Enroll(string speakerId, AudioBuffer audio, CancellationToken cancellationToken);
    /// <summary>Identifies against enrollments; null ID means unknown. Throws OperationCanceledException on cancellation.</summary>
    public abstract SpeakerMatch Identify(AudioBuffer audio, CancellationToken cancellationToken);
    /// <summary>Scores a pair of samples without enrollment; throws OperationCanceledException on cancellation.</summary>
    public abstract double Verify(AudioBuffer first, AudioBuffer second, CancellationToken cancellationToken);
    /// <summary>Removes an enrollment, returning false if absent. Throws OperationCanceledException on cancellation.</summary>
    public abstract bool Remove(string speakerId, CancellationToken cancellationToken);
}

/// <summary>Combines recognition, timing and speaker assignment with one audio clock.</summary>
public abstract class TranscriptionPipeline
{
    /// <summary>Processes in-memory audio; throws OperationCanceledException on cancellation.</summary>
    public abstract Transcript Transcribe(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default);
    /// <summary>Yields final segments from bounded audio chunks; throws OperationCanceledException on cancellation.</summary>
    public abstract IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language = null, CancellationToken cancellationToken = default);
}
