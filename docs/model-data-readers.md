# Model-data and document readers

This is the inventory of file-reading code owned by OneSpeech. The mutation targets are development tools; [measured results](results.md) report the completed campaigns separately.

| Reader | Model-supplied or caller-supplied data | Mutation target |
|---|---|---|
| Transcript JSON | Versioned document, including nested model metadata, words and turns | `transcript` |
| Catalog JSON | Model entries, files, capabilities, licence and tier metadata | `catalog` |
| Audio input adapter / G.711 WAV | Encoded stream headers and bounded decode reads | `wav`, `audio` |
| Tensor checkpoint | ZIP directory, inert pickle tensor index, storage views | `checkpoint`, `pickle` |
| Safetensors | Header JSON, names, shapes, dtypes, offsets and payload reads | `safetensors` |
| SentencePiece character vocabulary | Bounded protobuf model data | `sentencepiece` |
| Moonshine tokenizer / configuration | BPE JSON and offline/streaming configuration JSON | `moonshine` |
| Parakeet vocabulary / phrase tokenizer | Tokenizer JSON, merge data, precompiled normalization map | `parakeet` |
| Whisper tokenizer | `vocab.json`, `added_tokens.json`, UTF-8 `merges.txt` | `whisper-tokenizer` |
| Qwen tokenizer | `vocab.json`, `tokenizer_config.json`, UTF-8 `merges.txt` | `qwen-tokenizer` |
| Whisper configuration / generation controls | `config.json`; `generation_config.json` token IDs, language IDs, suppression and alignment heads | `whisper-config`, `whisper-generation` |
| Qwen ASR / forced-aligner configuration | Nested audio/text `config.json` | `qwen-config` |
| Nemotron ASR prompt configuration | `processor_config.json` language-to-prompt dictionary | `nemotron-processor` |
| Converter provenance | Bounded `upstream.txt`, known publisher revision and digest records | `provenance` |
| RTTM | Speaker intervals read by the evaluation tool, not core | `rttm` |

Whisper and Qwen byte vocabularies must contain the 256 byte symbols. IDs and file sizes are bounded: Whisper uses fewer than 60,000 IDs and at most 8 MiB per tokenizer file; Qwen uses fewer than 200,000 IDs and at most 16 MiB per file. A nonempty special-token spelling is required so token recognition always advances. Whisper's unused ordinary empty token is allowed and decodes to empty text. Merge references must be consistent with the declared vocabulary.

For the Whisper and Qwen tokenizer/configuration readers, syntactically malformed JSON throws `JsonException`. Incorrect JSON shapes, missing required properties, invalid value types and inconsistent token/configuration data are refused with `InvalidDataException`, rather than leaking JSON getter or collection exceptions. Provenance, integrity checks and the Nemotron processor reader use `ModelIntegrityException`; an unsupported Moonshine checkpoint retains its typed capability refusal. Filesystem failures remain I/O errors. Cancellation and public tokenizer text/ID argument errors retain their documented behavior.

Generation metadata token IDs must fit the configured vocabulary; alignment pairs must fit decoder layers/heads. Language entries are limited to 256, suppression entries to the vocabulary size, and selected heads to the decoder's layer/head product. The existing 1 MiB JSON bound also applies.

The two byte-BPE targets mutate each component file against deterministic synthetic companion files, then exercise encoding and decoding. Configuration targets parse without loading model graphs. The provenance target exercises Whisper, Moonshine and both Qwen roles; companion files intentionally stop a valid provenance record before neural graph loading. No checkpoint is downloaded or executed by these targets.

OneModel-format/ONNX graph parsing belongs to OneModel. Codec and container internals belong to OneAudio and OneMedia; the audio targets instrument OneSpeech's input adapter, not those dependencies. Caption imports consume OneMedia cue objects rather than implementing another subtitle-file parser. Binary speaker embeddings are not a library persistence format. The library does not read or persist a speaker-enrolment database.
