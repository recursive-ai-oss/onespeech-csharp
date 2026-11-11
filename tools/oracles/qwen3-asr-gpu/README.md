# Qwen3-ASR reference on a GPU

Same black-box reference as `../qwen3-asr` (qwen-asr 0.0.6, fp32, eager attention, English, no context), run on a CUDA device with TF32 disabled. `../qwen3-asr/run.py --device cuda [--batch N]` is the driver; it is shared with the CPU oracle.

    uv sync --frozen --project tools/oracles/qwen3-asr-gpu
