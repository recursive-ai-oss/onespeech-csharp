# Moonshine

The registered English checkpoints are `moonshine-ai/moonshine-tiny`,
`moonshine-base`, `moonshine-streaming-tiny`, `moonshine-streaming-small` and
`moonshine-streaming-medium` (all IDs use the `moonshine-ai/` publisher prefix).
They are pinned, ungated MIT checkpoints. Other checkpoints, including the
legacy Community License models, are refused. The catalog records upstream
and converted file hashes; local conversion uses only verified publisher data.
Models load as CPU fp32 OneModel graphs. No Python or native inference runtime
is required by the library.

```csharp
await foreach (Segment segment in Speech.TranscribeStream(
    "lecture.flac", model: "moonshine-ai/moonshine-base"))
    Console.WriteLine(segment.Text);

// pcm is an application-owned IAsyncEnumerable<AudioBuffer>, mono 16000 Hz.
await foreach (LiveEvent update in Speech.Listen(
    pcm, model: "moonshine-ai/moonshine-streaming-tiny"))
    if (update is PartialEvent partial) Console.Write(partial.Segment.Text);
```

`Speech.Listen` defaults to streaming-tiny. Offline tiny/base use VAD-endpointed
utterance mode and emit finals only. The file-transcription default is unchanged;
select a Moonshine model explicitly for file recognition. The file-streaming
API awaits model I/O and yields segments asynchronously; decoding/inference use
the synchronous CPU APIs, without background-task wrappers.

## Recognition contract

`MoonshineRecognizer.Load(directory, options)` opens verified converted graphs,
`tokenizer.json`, `config.json` and three-line `upstream.txt` provenance. Loading
checks file hashes against the catalog. The low-level recognizer accepts finite
mono 16 kHz PCM and utterances no longer than 60 s; use the bounded VAD pipeline
for longer recordings. Empty audio returns an empty transcript. Nonempty inputs
shorter than the frontend minimum (895 samples offline, 80 streaming) are
refused. Other rates/channels and nonfinite samples are refused with
`UnsupportedAudioException`.

Decoding is fp32 greedy, starting with BOS ID 1 and stopping at EOS ID 2 or the
new-token budget. Defaults are one inference thread and 6.5 new tokens per
second of real input, excluding BOS. A null language hint selects the fixed
English task; it does not run language identification. Other languages,
translation, contextual biasing and language detection are not provided.

Words preserve unknown timing and confidence as null, with `TimingMethod.None`.
VAD segment bounds are not word boundaries. These checkpoints do not produce
native word timestamps. Speaker assignment is a separate operation.

## Frontends and numerics

The offline architecture follows Jeffries et al., *Moonshine*, arXiv
2410.15608v2, §3.1: 16 kHz raw waveform, valid convolution kernels 127/7/3 with
strides 64/3/2, tanh and group normalization after the first convolution, then
GELU after each remaining convolution. Encoder attention and decoder
self-attention use rotary embeddings. Decoder feed-forward blocks use a
value half followed by a SiLU-gated half.

The streaming architecture follows the publisher's *Moonshine Streaming*
paper, §§3.1–3.4: 80-sample frames, per-frame population CMVN with variance
regularizer 1e-6, learned asinh compression, linear/SiLU, two causal
kernel-5/stride-2 convolutions. Only the first convolution has a following SiLU.
For an incomplete 80-sample frame, the released feature extractor pads the
waveform and the public model masks that entire frame before convolution.
We retain the padded frame count and zero the incomplete frame, as confirmed
by frontend and first-logit comparisons on 321 samples; silently reducing
the encoder frame count would give different output.
The encoder is position-free; learned positions and an optional projection
adapt its output to the decoder.

Details checked against released Transformers 5.22.2 public outputs include
adjacent-pair rotary embeddings, the decoder gate ordering, LayerNorm's
`1 + gamma` encoder scale and epsilon 1e-5, and local-window endpoints. For
window `(16,4)`, keys satisfy `query - 15 <= key <= query + 3`; `(16,0)` includes
the current key with no future keys. First/last two layers use lookahead;
intermediate layers are causal.

The fp32 comparison criteria are maximum scaled error
`abs(actual - reference) / max(1, abs(reference)) <= 1e-4` for the offline
frontend, final encoder output and decoder logits, and maximum absolute error
1e-5 for streaming frontend features. These are fast-arithmetic criteria, not
bitwise tensor equality. Tokenizer IDs and decoded strings must match exactly.

## Live state

Streaming checkpoints emit replacing partials no more often than once per
200 ms of accepted audio, after minimum speech is established. Finalization
uses the shared VAD endpoint policy in [Live sessions](live-sessions.md).

Encoder prefixes whose right context is complete are retained. Only a bounded
suffix is re-encoded: its left context covers every layer's 15 past frames and
three frontend frames, and its provisional suffix covers 12 lookahead frames.
This does not repeatedly encode the growing whole utterance. A new endpoint
resets session-local caches. Each new audio hypothesis starts at BOS, because changed encoder memory can
change earlier decoder states. Within that hypothesis, the decoder evaluates
one token per call and retains each layer's self-attention keys and values.
Cross-attention keys/values are projected once per audio hypothesis, not once
per token. Cache planes are packed into one tensor to keep graph bindings small.

Live sessions preallocate and warm their working state before accepting PCM.
The encoder uses a fixed receptive-field window; decoder cross-attention uses
masked frame buckets of 128/256/512/1024/1536 frames. Self-attention cache capacity
is a power of two covering the configured token budget over the shared 30-second
VAD horizon. Unused keys are masked; padded encoder queries remain finite and
cannot affect real frames. These bounds do not change offline recognition's
60-second limit. Session creation includes the warm-up work.

Immutable emitted events, their text and word payloads belong to the caller and
are never pooled or mutated. They are measured separately from working-state
allocation. At the validated 100 ms push size, an emitting push also needs no
new working buffers. Backend runtime allocations are reported separately; zero
working-state allocation is not a claim of zero total process allocation while
the backend still allocates.

References: [offline paper](https://arxiv.org/abs/2410.15608),
[streaming paper](https://download.moonshine.ai/docs/moonshine_streaming_paper.pdf).
The exact checkpoint revisions and model-card URLs are in the catalog.
