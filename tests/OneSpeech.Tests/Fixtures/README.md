# Speech recognition for .NET — reference observations

OneSpeech is unaffiliated with other products or companies named OneSpeech.

`tone-80.json` and `tone-128.json` contain output observed from openai-whisper 20250625 on the deterministic synthetic signal described in each file.

`whisper-tokenizer-cases.json` contains synthetic text cases. `whisper-tokenizer-reference.json` contains observed token IDs and decoded text from transformers 5.22.2, using the pinned Whisper-tiny tokenizer data, with no automatic prefix tokens, no cleanup and explicit timestamp decoding. It contains no implementation source.

The tiny transcription smoke pins the model's observed output on the attributed LibriSpeech clip, including its recognition errors. It is a regression/parity observation, not the corpus's ground-truth transcript. Models are fetched separately; set `ONESPEECH_WHISPER_TINY` to the verified converted model directory to run the model tests.
