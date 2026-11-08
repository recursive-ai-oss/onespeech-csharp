# Integer inference

The fp32 defaults are unchanged. The following explicit catalog IDs select locally
converted, mixed-int8 graphs:

- `openai/whisper-large-v3-turbo-int8`
- `openai/whisper-small-int8`
- `moonshine-ai/moonshine-streaming-tiny-int8`

For example:

```csharp
var transcript = Speech.Transcribe("recording.wav", language: "en",
    model: "openai/whisper-large-v3-turbo-int8");
```

The variants use the same pinned publisher weights, configuration and tokenizer
files as their fp32 counterparts. Conversion is deterministic and runs in the
local model cache. Each converted file has a pinned size and SHA-256 in the
catalog. The model ID in `ModelInfo` identifies the int8 variant; its source and
revision continue to identify the original publisher checkpoint. No public API
or transcript JSON schema changes are required.

## Numerical scheme

Whisper constant-weight linear projections, including vocabulary projection, use
signed int8 weights with one scale per output channel. Moonshine quantizes only
the decoder MLPs and vocabulary projection; its acoustic encoder, attention and
cross-attention projections remain fp32 to protect recognition quality. For each channel, the scale is
`max(abs(weight)) / 127`, or 1 for an all-zero channel. Values are divided by the
scale, rounded to nearest with ties to even, and clamped to [-127, 127]. The stored
weight is transposed to the matrix product's input/output layout. No development
or evaluation data is used to choose scales.

Activations at quantized linear products use ONNX
`DynamicQuantizeLinear`: the range includes zero, uses 255 intervals, and an
all-zero input has scale 1 and zero point 0.

`MatMulInteger` accumulates into int32. Its fp32 accumulator is multiplied by the
product of activation and weight scales before adding any fp32 bias. No collected
activation statistics are involved. The supported checkpoints' linear input
widths are below OneModel's integer accumulation bound.

Convolutions, embedding lookup, normalization, attention score/value products,
residuals and nonlinearities remain fp32. These are mixed-int8 models, not
all-integer networks. Quantization changes logits and can change decoded tokens;
fp32 token identity is not promised. Integer inference also does not guarantee a
speed improvement on every CPU or thread count.
