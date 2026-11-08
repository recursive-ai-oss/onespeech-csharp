# Gated models and segmentation

Gated entries are tier 2, explicitly named, never defaults or mirrors. The
catalog pins publisher revisions and each file's SHA-256. `ModelStore.PullAsync`
takes an ephemeral API token, falling back to `ONESPEECH_MODEL_TOKEN` only when
the API argument is null. It checks gate credentials even for cached files.
A missing or whitespace token throws `ModelTokenRequiredException` before I/O.

The HTTP transport sends Bearer credentials only on the initial request to the
catalogued publisher file URL. Mirrors and every redirect target, including
same-host redirects, receive no credentials. Redirects must remain HTTPS, without
userinfo or custom ports, and stop after five hops. HTTP 401/403 responses throw
`ModelTokenRejectedException`, unless the publisher's `X-Error-Code` is
`GatedRepo`, which throws `ModelGateNotAcceptedException`. A server that does not
distinguish a gate refusal from a rejected credential cannot be classified more
precisely. Response bodies, error-message headers, credentials and signed URLs
never enter library exceptions, cache files or `ModelInfo`. No response body is
parsed to guess gate state. Custom `ModelSource` implementations are responsible
for the same confidentiality rules and are trusted host code.

The clustering segmentation slot is the `VoiceActivityDetector` argument of
`ClusteringDiarizer`. A neural segmenter can implement that role and return the
union of its speech intervals for each bounded analysis block. The engine
validates intervals, extracts embeddings only inside those intervals, and never
claims overlap detection from a union-of-speech mask. Detectors are borrowed,
not disposed by the engine.

`PyannoteSegmentation.Load` names the two gated catalog entries and refuses
missing credentials with `ModelTokenRequiredException`. With a token it currently
throws `FeatureUnavailableException`: the pyannote network has not been
implemented or evaluated. It does not download unsupported weights. Explicit
prefetch remains available through `ModelStore.PullAsync`. Until network support
is available, use Silero or a caller-provided speech-interval detector; no guessed
network or overlap result is substituted.
