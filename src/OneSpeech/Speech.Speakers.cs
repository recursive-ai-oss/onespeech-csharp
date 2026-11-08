namespace OneSpeech;

public static partial class Speech
{
    /// <summary>Enrolls or replaces a case-sensitive nonblank name in process memory only. Obtain speaker consent and comply with local biometric laws. Samples are downmixed to mono 16 kHz, nonempty, at most 60 seconds. Language selects English CAM++ by default or Chinese CAM++ for zh; model explicitly selects either catalog ID. Missing weights download on first use. Throws ArgumentException, typed audio/model/capability errors or IOException. ClearSpeakers releases enrollments and loaded speaker models. Never persists embeddings.</summary>
    public static void Enroll(string name, string path, string? language = null, string? model = null) => Enroll(name, path, CancellationToken.None, language, model);
    /// <summary>Enrolls with cancellation at decoding blocks and model calls; failed extraction preserves existing enrollment. Consent, limits and errors follow Enroll; cancellation throws OperationCanceledException.</summary>
    public static void Enroll(string name, string path, CancellationToken cancellationToken, string? language = null, string? model = null) => Pipeline.SpeakerServices.Enroll(name, path, language, model, cancellationToken);
    /// <summary>Identifies the best enrolled name, or null for unknown, with raw cosine score (not a probability). Uses the calibrated inclusive threshold for the selected model; models have separate enrollment spaces. Empty registry returns negative infinity. Obtain consent and comply with local biometric laws; limits, defaults and errors follow Enroll.</summary>
    public static SpeakerMatch Identify(string path, string? language = null, string? model = null) => Identify(path, CancellationToken.None, language, model);
    /// <summary>Identifies with cancellation. Consent, limits and errors follow Identify; cancellation throws OperationCanceledException.</summary>
    public static SpeakerMatch Identify(string path, CancellationToken cancellationToken, string? language = null, string? model = null) => Pipeline.SpeakerServices.Identify(path, language, model, cancellationToken);
    /// <summary>Scores two files without enrollment. Returns raw cosine in [-1,1], not a probability or an automatic verification decision. Obtain consent and comply with local biometric laws. Samples, defaults and errors follow Enroll; no embedding is persisted.</summary>
    public static double Verify(string firstPath, string secondPath, string? language = null, string? model = null) => Verify(firstPath, secondPath, CancellationToken.None, language, model);
    /// <summary>Verifies a pair with cancellation. Consent, limits and errors follow Verify; cancellation throws OperationCanceledException.</summary>
    public static double Verify(string firstPath, string secondPath, CancellationToken cancellationToken, string? language = null, string? model = null) => Pipeline.SpeakerServices.Verify(firstPath, secondPath, language, model, cancellationToken);
    /// <summary>Enrolls from a readable encoded-audio stream, borrowed and left open at its consumed position. Encoded input is temporarily spooled for decoding and deleted on completion or failure; embeddings are never persisted. Obtain consent and comply with local biometric laws. Limits/defaults/errors follow Enroll; cancellation throws OperationCanceledException.</summary>
    public static ValueTask EnrollAsync(string name, Stream stream, string? language = null, string? model = null) => EnrollAsync(name, stream, CancellationToken.None, language, model);
    /// <summary>Enrolls a borrowed encoded stream with cancellation during I/O, decoding and extraction. Consent, spooling, limits and errors follow EnrollAsync.</summary>
    public static ValueTask EnrollAsync(string name, Stream stream, CancellationToken cancellationToken, string? language = null, string? model = null) => Pipeline.SpeakerServices.EnrollAsync(name, stream, language, model, cancellationToken);
    /// <summary>Identifies a borrowed readable encoded stream; leaves it open. Obtain consent and comply with local biometric laws. Temporary spooling, limits and errors follow EnrollAsync; scores and unknown handling follow Identify.</summary>
    public static ValueTask<SpeakerMatch> IdentifyAsync(Stream stream, string? language = null, string? model = null) => IdentifyAsync(stream, CancellationToken.None, language, model);
    /// <summary>Identifies a borrowed stream with cancellation. Consent, limits and errors follow IdentifyAsync.</summary>
    public static ValueTask<SpeakerMatch> IdentifyAsync(Stream stream, CancellationToken cancellationToken, string? language = null, string? model = null) => Pipeline.SpeakerServices.IdentifyAsync(stream, language, model, cancellationToken);
    /// <summary>Scores two borrowed readable encoded streams, left open. Obtain consent and comply with local biometric laws. Temporary spooling, limits and errors follow EnrollAsync; score semantics follow Verify. No embedding is persisted.</summary>
    public static ValueTask<double> VerifyAsync(Stream first, Stream second, string? language = null, string? model = null) => VerifyAsync(first, second, CancellationToken.None, language, model);
    /// <summary>Scores two borrowed streams with cancellation. Consent, limits and errors follow VerifyAsync.</summary>
    public static ValueTask<double> VerifyAsync(Stream first, Stream second, CancellationToken cancellationToken, string? language = null, string? model = null) => Pipeline.SpeakerServices.VerifyAsync(first, second, language, model, cancellationToken);
    /// <summary>Clears all process-wide speaker enrollments and releases their loaded models. Idempotent; waits for active speaker operations. Clearing managed storage is best effort, not protection against memory dumps. Obtain consent and comply with local biometric laws before reenrollment.</summary>
    public static void ClearSpeakers() => ClearSpeakers(CancellationToken.None);
    /// <summary>Clears speaker state with cancellable lock acquisition; cancellation leaves state unchanged and throws OperationCanceledException. Consent and storage caveats follow ClearSpeakers.</summary>
    public static void ClearSpeakers(CancellationToken cancellationToken) => Pipeline.SpeakerServices.Clear(cancellationToken);
}
