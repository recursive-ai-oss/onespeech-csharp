namespace OneSpeech;

/// <summary>Base for actionable speech processing failures.</summary>
public class OneSpeechException : Exception
{
    /// <summary>Creates a failure with a human-readable fix or explanation.</summary>
    public OneSpeechException(string message) : base(message) { }
    /// <summary>Creates a failure preserving its underlying cause.</summary>
    public OneSpeechException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>The requested model is absent; the message names the model to fetch.</summary>
public sealed class ModelNotFoundException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public ModelNotFoundException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public ModelNotFoundException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>A model file failed its expected size or SHA-256 check.</summary>
public sealed class ModelIntegrityException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public ModelIntegrityException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public ModelIntegrityException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>The publisher requires a user token or gate acceptance.</summary>
public sealed class ModelTokenRequiredException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public ModelTokenRequiredException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public ModelTokenRequiredException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>A requested capability is unsupported and was refused.</summary>
public sealed class CapabilityNotSupportedException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public CapabilityNotSupportedException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public CapabilityNotSupportedException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>Audio is corrupt or its format is unsupported.</summary>
public sealed class UnsupportedAudioException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public UnsupportedAudioException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public UnsupportedAudioException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>Transcript JSON is malformed, unsupported, invalid or exceeds reader bounds.</summary>
public sealed class InvalidTranscriptJsonException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public InvalidTranscriptJsonException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public InvalidTranscriptJsonException(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>A feature has not been installed or implemented; the message describes the missing component.</summary>
public sealed class FeatureUnavailableException : OneSpeechException
{
    /// <summary>Creates a typed failure with an explanation and required fix.</summary>
    public FeatureUnavailableException(string message) : base(message) { }
    /// <summary>Creates a typed failure preserving the underlying cause.</summary>
    public FeatureUnavailableException(string message, Exception innerException) : base(message, innerException) { }
}
