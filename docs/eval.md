# Evaluation

Speech recognition for .NET. These tools score every system with the **same** reference text and normalizer. They are not part of the runtime package. The project is unaffiliated with other products or companies named OneSpeech.

## Normalizer version 1

`en` is our English scoring normalizer, written from the descriptions in Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, arXiv:2212.04356, Appendix C and §4.4, and *The Open ASR Leaderboard*, arXiv:2510.06961, §2.3. It is **not** a claim of byte-exact equivalence to either project's software. Report `en/1` in every score row; changing these rules requires a new version and rescoring both systems.

| Rule | Source and concrete policy |
|---|---|
| Unicode and casing | Appendix C: NFKC, invariant lowercase; English additionally decomposes diacritics and deletes combining marks. |
| Annotations | Appendix C rules 1–2: remove matched square-bracket/parenthetical phrases. One non-nested pass; unmatched delimiters become punctuation separators. |
| Fillers | Appendix C rule 3: remove whole words `hmm mm mhm mmm uh um`. No other fillers are inferred. The leaderboard describes filler removal but does not supply a list. |
| Contractions | Appendix C rules 4–5: unify curly apostrophes, remove whitespace before apostrophes; expand `n't`, `'re`, `'ve`, `'ll`, `'m`, `'d` (would). Explicit irregular forms: can't/cannot→can not, won't→will not, shan't→shall not, ain't→is not, let's→let us. A finite list of pronoun/demonstrative `'s` forms expands to “is”. Other possessive `'s` remains a separate `s`. Informal gonna/wanna/gotta/imma expand to going to/want to/got to/i am going to. Ambiguity is resolved lexically, not from context. |
| Punctuation | Appendix C rules 6–8,11: remove digit-internal commas; punctuation/symbols become spaces (hyphens split words), except a decimal point between digits, currency prefix `$ £ €` before digits and `%` after digits. Diacritics are removed, not separated. |
| Numbers | Appendix C rule 9; leaderboard §2.3: zero through nineteen, tens through ninety, hundred, thousand/million/billion use Arabic digits. Lower units add, scales multiply the current group; “and” is consumed before a following number word. `point` plus single-digit words forms a decimal. Digit literals can take scales (`$68 million`→`$68000000`). No year/phone grouping, fractions, negative-word interpretation, or semantic number disambiguation. Unsupported words remain words. |
| Ordinals | Our finite extension of number normalization: first through twelfth, twentieth, thirtieth, hundredth; compound values take st/nd/rd/th with 11–13 exception. Other ordinal words remain words. |
| Currency | Appendix C rule 9: a numeric value followed by dollar(s), pound(s), euro(s) becomes a prefixed numeric token. No cents composition, exchange rates or regional currency inference. |
| Spelling | Appendix C rule 10: a finite explicit table, not a dictionary dependency. colour(s), favour, favourite, honour, labour, neighbour, centre(s), theatre, metre, litre, organise(d), recognise, realise, travelling, cancelled, grey, defence, licence map to their American forms. Other spellings remain unchanged. |
| Whitespace | Appendix C rule 12: collapse whitespace and trim. |

Repetitions and repairs are **preserved**. The leaderboard §2.2 discusses dataset transcriptions with those disfluencies removed; it does not specify an executable semantic repair algorithm. Deleting repeated words could conceal actual decoder errors. All finite tables above are project choices where the papers leave details unspecified, not transcriptions of a reference implementation's tables.

`basic/1`: NFKC, Unicode 17.0.0 default full case folding (CaseFolding.txt C and F mappings, no Turkic tailoring), replace Unicode M/S/P categories with spaces, collapse whitespace; retains precomposed accents. The generated case-fold table comes from the Unicode Character Database under Unicode License V3 (notice in `eval/OneSpeech.Eval/Unicode/LICENSE.txt`). CER counts Unicode scalar values after normalization, omitting whitespace; no grapheme segmentation. Use CER, rather than whitespace WER, for text without word separators. The sources describe basic multilingual normalization but do not supply language-specific linguistic rules.

## Metrics

WER = (substitutions + deletions + insertions) / reference word count. Corpus WER is the ratio of summed counts, **not** the mean of utterance rates. CER uses the same formula over scalar values. Levenshtein cost is one per edit, zero for an exact match. Ties prefer diagonal, then deletion, then insertion. Counts use linear-row memory; optional alignment traces are bounded to 16 million cells. An empty reference has undefined rate (`null`), but its insertion count still contributes to a corpus with a nonempty reference. Extra and missing IDs are errors; a failed decode must be an explicit empty hypothesis.

Bootstrap: sort utterances by ordinal ID, draw N utterances with replacement for each replicate, summing counts before dividing. Two systems use the **same** draw. Difference is first WER minus second WER, in rate units (multiply by 100 for percentage points). Default 1000 resamples, seed 1729; fewer than 1000 are refused. SplitMix64 selects an index by unsigned remainder modulo N. Sort replicates and report percentile bounds at floor(0.025×(B−1)), ceil(0.975×(B−1)). These are descriptive percentile intervals, not hypothesis-test p-values. Fewer than two utterances or any zero-length normalized reference makes CIs unavailable (avoids undefined zero-denominator resamples); the corpus score is still reported.

Verdict: `within-noise` iff the paired difference CI includes zero. Only if a CI is unavailable, absolute difference ≤0.005 (0.5 percentage points) gives `within-fallback`. Otherwise `different`; undefined corpus rates or no second system give `not-comparable`. A small difference with a computable CI excluding zero does **not** use the fallback.

Timing: align **word objects** by their normalized text with the same Levenshtein tie policy. Compare only exactly matching diagonal pairs; do not interpolate unmatched tokens or split multi-token expansions into invented timed words. Report matched, timed, missing counts; mean and median absolute start/end error in milliseconds and separate start/end fractions within inclusive 50/100 ms. Missing intervals are counted, not assigned a zero error. Timing summaries appear per utterance in JSON; no identity or temporal nearest-neighbor matching is used.

## Files

UTF-8 JSON lines, one object per utterance:

```json
{"id":"1089-134686-0000","text":"Hello world","words":[{"text":"Hello","start":0.25,"end":0.65}]}
```

IDs must be nonblank and unique. Words are optional. Times are finite nonnegative seconds on the original file clock; both endpoints must be null or both numbers, end ≥ start. JSONL lines are limited to 2 Mi characters and JSON nesting to 32. An optional first record with `"type":"header"` holds reproducibility metadata and is ignored by the scorer. Records with missing text, duplicate IDs, malformed JSON or invalid word intervals fail scoring; no records are silently dropped. Input parsers are for local evaluation artifacts, not internet endpoints.

A single version-1 `Transcript` JSON document is also accepted through the core's validated reader, with the filename stem as utterance ID. See [Transcript JSON](transcript-json.md). It cannot stand in for multiple utterances without an explicit segmentation manifest.

Audio manifest for oracle drivers (not scoring inputs): JSONL `{"id":"...","audio":"/caller/selected/file.flac"}` in fixed order. Ranges are zero-based, start-inclusive/end-exclusive. Relative audio paths resolve from the process's working directory. Save the manifest and its hash with the row.

## Running a row

```sh
bash tools/scripts/fetch-librispeech.sh "$DATA/librispeech"
bash tools/scripts/fetch-earnings22.sh "$DATA/earnings22"
bash tools/scripts/install-oracles.sh "$ORACLES"
dotnet run --project eval/OneSpeech.Eval -- references --dataset librispeech --input "$DATA/librispeech/LibriSpeech/test-clean" --out refs.jsonl
dotnet run --project eval/OneSpeech.Eval -- references --dataset earnings22 --input "$DATA/earnings22/transcripts/nlp_references" --out earnings-refs.jsonl
```

LibriSpeech reads the text after the ID in each `*.trans.txt`. Earnings-22 reads column 1 of each pipe-separated `.nlp` row (eight columns, header required); punctuation/case metadata is not needed for normalized scoring. It also preserves the optional timing columns. Use original `nlp_references`, not force-aligned copies, so files do not duplicate IDs.

The fetch scripts verify existing archives/media without downloading them again. Text files are verified against Git blob IDs at pinned revision `c05ab6fd8b4b627d123c922a22a39e993dd37635`; SHA-256 and URL/licence records are written beside the cache. Existing hash mismatches fail instead of being overwritten. LibriSpeech is CC BY 4.0; Earnings-22 CC BY-SA 4.0, evaluation only, never committed or redistributed by these scripts. Extraction is from hash-verified archives. Delete the extraction marker to re-extract.

Oracle environments are separate `uv` projects with lockfiles and CPU torch. `run.py` uses only public APIs. Every system receives the same soundfile float32 decode, arithmetic channel mean, and librosa `soxr_hq` resampling to 16 kHz, with no amplitude normalization or ffmpeg. CPU/fp32, language English, task transcription, greedy decoding, no previous-text conditioning, and no stochastic fallback are explicit. No GPU support is offered. The output header includes the public transcribe signature defaults and DecodingOptions for OpenAI, and the model and explicitly supplied effective generation configurations for Transformers. Signature metadata is inspected without reading source or bytecode. These configurations are **not** a promise that long-form pipelines are equivalent. Transformers uses 30-second pipeline chunks; OpenAI uses its own long-form transcribe procedure. Consequently this driver is a reference sanity check, not an assertion of token identity.

The OpenAI driver calls `whisper.load_model("tiny", download_root=...)` when `--model tiny` is specified without `--checkpoint`. The released checkpoint is downloaded and verified by that public API; the driver records its SHA-256. An explicit `--checkpoint` can instead select an already downloaded release artifact. Use the HF tiny revision below for Transformers; do not substitute tiny.en.

```sh
wq run --class heavy --max 30m --cores 8 --label "onespeech reference tiny shard" -- \
  "$ORACLES/openai-whisper/bin/python" eval/drivers/run.py --oracle openai-whisper \
  --manifest manifest.jsonl --audio-root "$DATA/librispeech" --output tiny-0000.jsonl --model tiny --revision released-20250625 \
  --cache "$MODELS" --start 0 --end 100 --threads 8
wq run --class heavy --max 30m --cores 8 --label "onespeech reference transformers shard" -- \
  "$ORACLES/transformers/bin/python" eval/drivers/run.py --oracle transformers \
  --manifest manifest.jsonl --audio-root "$DATA/librispeech" --output hf-0000.jsonl --model openai/whisper-tiny \
  --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af --cache "$MODELS" --start 0 --end 100 --threads 8
dotnet run --project eval/OneSpeech.Eval -- score --ref refs.jsonl --hyp tiny.jsonl --hyp2 hf.jsonl --normalizer en --resamples 1000 --seed 1729 --out scores.json
```

Repeat disjoint ranges through 2620 for the complete test-clean set; concatenate records (headers are ignored by the scorer). Choose smaller shards if needed to remain under the job bound. Each output header records versions, settings, range and manifest hash; completed utterances are flushed and fsynced. Resume checks the exact header and IDs. A truncated final JSON line fails rather than silently skipping it: remove only that incomplete line before resuming. Keep all shard headers with the row. The faster-whisper variant uses `--oracle faster-whisper` and a pinned compatible publisher checkpoint, never assumes HF safetensors are CTranslate2 weights. Fetch scripts do not download models.

Output JSON includes per-utterance WER/CER counts, timing summaries, corpus counts, bootstrap bounds and verdict; rates are fractions, console output is percent. Keep model weight hashes, dataset hashes, oracle locks, machine CPU/ISA/core count, load, thread count, SDK/runtime and full command/job IDs with measured rows. These tools alone do not constitute a measured quality claim.

### Fixed targeted slice

`eval/manifests/librispeech-test-clean-first100.jsonl` pins the first 100 test-clean utterance IDs in ordinal order, their relative audio paths and each FLAC's SHA-256. It contains metadata only, not audio or reference text. Regenerate it from the verified cache with `eval/drivers/prepare_slice.py`; that script also writes a caller-selected reference JSONL outside the repository. Pass the dataset cache as `--audio-root` and use `--start 0 --end 100`. The driver checks manifest audio hashes before decoding. This small sanity check does not replace full-corpus evaluation.

The fixed slice was checked with openai-whisper 20250625 (`tiny`, original MIT release artifact) and Transformers 5.22.2 (`openai/whisper-tiny`, revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`). Both used fp32 CPU, greedy decoding, temperature 0 and English. Original `tiny.pt` SHA-256: `65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9`; it is a different artifact from the HF safetensors. With `en/1`, 1000 paired bootstrap resamples and seed 1729:

| System | Errors / reference words | WER | 95% CI |
|---|---:|---:|---:|
| OpenAI released tiny | 143 / 2357 | 6.067% | [4.932%, 7.300%] |
| HF tiny | 141 / 2357 | 5.982% | [4.852%, 7.231%] |

Paired difference (OpenAI minus HF): **0.085 percentage points**, 95% CI **[−0.626, 1.077]**, verdict **within-noise**. This is a small correctness check, not a full-corpus quality claim or a performance comparison. The run used Linux x64 on an AMD Ryzen 9 7950X3D (16 cores / 32 logical CPUs), eight oracle threads, torch 2.15.0+cpu, numpy 2.5.4, soundfile 0.13.1 and librosa 0.11.0; no GPU. Full per-utterance hypotheses and resolved decoding headers are evaluation artifacts, not embedded library data.
