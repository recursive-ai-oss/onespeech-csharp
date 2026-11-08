"""Documented Transformers processor geometry; no model code is inspected."""
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import json
import sys


def main():
    import transformers
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
    print(json.dumps({"transformers": transformers.__version__, "processor": type(processor).__name__}), flush=True)
    for right in [0, 1, 3, 6, 13]:
        try:
            processor.set_num_lookahead_tokens(right)
            print(json.dumps({"right_context": right,
                              "first_samples": processor.num_samples_first_audio_chunk,
                              "first_mel_frames": processor.num_mel_frames_first_audio_chunk,
                              "samples_per_chunk": processor.num_samples_per_audio_chunk,
                              "mel_frames_per_chunk": processor.num_mel_frames_per_audio_chunk,
                              "latency_ms": processor.streaming_latency_ms}), flush=True)
        except Exception as error:
            print(json.dumps({"right_context": right, "exception": type(error).__name__, "message": str(error)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"exception": type(error).__name__, "message": str(error)}), file=sys.stderr)
        sys.exit(1)
