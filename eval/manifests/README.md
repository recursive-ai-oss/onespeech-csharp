# Fixed reference slice

`librispeech-test-clean-first100.jsonl` contains the first 100 LibriSpeech test-clean IDs in ordinal order, relative FLAC paths and SHA-256 hashes. No audio or transcript text is included. Regenerate with `eval/drivers/prepare_slice.py` from the hash-verified test-clean archive; see [evaluation](../../docs/eval.md).

Source: [LibriSpeech, OpenSLR 12](https://www.openslr.org/12/), CC BY 4.0. Attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, *LibriSpeech: an ASR corpus based on public domain audio books* (2015). Metadata selection and content hashing are ours; the corpus is unchanged.
