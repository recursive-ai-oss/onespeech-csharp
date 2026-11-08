"""Exact IDs and decoding through Transformers' public tokenizer API on the same local data."""
import warnings
warnings.simplefilter("ignore")  # Do not let dependency warnings print installed-module source lines.
import argparse
import json
from pathlib import Path
import sys
from transformers import WhisperTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = WhisperTokenizerFast.from_pretrained(args.directory, local_files_only=True, clean_up_tokenization_spaces=False)
    values = []
    for text in json.loads(args.cases.read_text()):
        ids = tokenizer.encode(text, add_special_tokens=False)
        values.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, decode_with_timestamps=True)})
    args.output.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
