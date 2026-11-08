"""Scalar comparison of released weights; no converted weight files are produced."""
import warnings
warnings.simplefilter("ignore")  # Do not let dependency warnings print installed-module source lines.
import argparse
import json
import struct
import sys
import numpy as np
import torch
import whisper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("safetensors")
    parser.add_argument("weights")
    args = parser.parse_args()
    model = whisper.load_model(args.model, device="cpu", download_root=args.weights).float()
    with open(args.safetensors, "rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(length))
    mappings = [("encoder.positional_embedding", "model.encoder.embed_positions.weight"), ("encoder.conv1.weight", "model.encoder.conv1.weight"), ("encoder.blocks.0.attn.query.weight", "model.encoder.layers.0.self_attn.q_proj.weight"), ("encoder.ln_post.weight", "model.encoder.layer_norm.weight")]
    state = model.state_dict()
    for official, hf in mappings:
        info = header[hf]
        array = np.memmap(args.safetensors, mode="r", dtype={"F32": "<f4", "F16": "<f2"}[info["dtype"]], offset=8 + length + info["data_offsets"][0], shape=tuple(info["shape"]))
        reference = state[official].numpy()
        error = np.abs(array.astype("float32") - reference)
        print(official, "max_abs", float(error.max()), "mean_abs", float(error.mean()), "equal", bool(np.array_equal(array, reference)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
