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
public abstract class VoiceActivityDetector
{
    /// <summary>Model provenance for this detector.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Returns speech intervals; empty audio yields no intervals. Throws OperationCanceledException on cancellation.</summary>
    public abstract ImmutableArray<SpeechInterval> Detect(AudioBuffer audio, CancellationToken cancellationToken);
}

/// <summary>Recognizes in-memory audio synchronously. Live sessions are a separate future contract.</summary>
public abstract class SpeechRecognizer
{
    /// <summary>Loaded model provenance.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Requests outside this declaration throw CapabilityNotSupportedException.</summary>
    public abstract Capabilities Capabilities { get; }
    /// <summary>Transcribes audio, with a language tag or null for model detection. Throws OperationCanceledException on cancellation.</summary>
    public abstract Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken);
    /// <summary>Yields final immutable segments in audio order, using bounded chunks. Throws OperationCanceledException on cancellation.</summary>
    public abstract IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken);
    /// <summary>Detects a language tag; throws CapabilityNotSupportedException if unsupported and OperationCanceledException on cancellation.</summary>
    public abstract string DetectLanguage(AudioBuffer audio, CancellationToken cancellationToken);
}

/// <summary>Aligns written text with in-memory audio without inventing failed timings.</summary>
public abstract class Aligner
{
    /// <summary>Loaded model provenance.</summary>
    public abstract ModelInfo Model { get; }
    /// <summary>Supported languages and timing features.</summary>
    public abstract Capabilities Capabilities { get; }
    /// <summary>Aligns text to audio. Null language requests detection. Throws OperationCanceledException on cancellation.</summary>
    public abstract Transcript Align(AudioBuffer audio, string text, string? language, CancellationToken cancellationToken);
}

/// <summary>Assigns speaker turns to in-memory audio; implementations declare whether overlap is detected.</summary>
public abstract class Diarizer
{
    /// <summary>Supported speaker count and other limits.</summary>
    public abstract Capabilities Capabilities { get; }
    /// <summary>Returns turns on the input audio clock; throws OperationCanceledException on cancellation.</summary>
    public abstract ImmutableArray<Turn> Diarize(AudioBuffer audio, CancellationToken cancellationToken);
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
    public abstract Transcript Transcribe(AudioBuffer audio, string? language, CancellationToken cancellationToken);
    /// <summary>Yields final segments from bounded audio chunks; throws OperationCanceledException on cancellation.</summary>
    public abstract IEnumerable<Segment> TranscribeStream(AudioBuffer audio, string? language, CancellationToken cancellationToken);
}
