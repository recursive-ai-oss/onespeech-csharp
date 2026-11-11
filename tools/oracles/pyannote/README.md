# pyannote CPU reference

This evaluation-only environment pins pyannote.audio 4.0.7 (MIT, Copyright
2020 CNRS) and CPU Torch/torchaudio 2.11.0. Use `uv sync --frozen` in this
directory. The scripts call only the model card's documented public APIs,
with local authorized snapshots and `HF_HUB_OFFLINE=1`. They never fetch models
or accept tokens. Dependency source and installed module files are not inputs.

From the repository root:

```sh
python tools/oracles/pyannote/segmentation.py SNAPSHOT RESULTS --corpus LIBRISPEECH_TEST_CLEAN
python tools/oracles/pyannote/diarize.py COMMUNITY_SNAPSHOT AUDIO.wav OUTPUT.rttm
```

The segmentation probe writes three analytic signals plus the first five
ordinal LibriSpeech test-clean clips, padded/truncated to 10 seconds, and their
589 × 7 frame log-probabilities. Inputs and outputs remain evaluation artifacts,
not redistributable fixtures. `tensors.json` records names and shapes only, not
weights. Compare with `OneSpeech.Convert pyannote-probe CHECKPOINT CACHE RESULTS`.
The diarization runner writes overlapping, not exclusive, RTTM turns and skips
completed outputs so file-level shards can resume. Score both sides with
`onespeech-eval der --ref TRUTH.rttm --hyp OUTPUT.rttm --out METRICS.json`.

Model provenance: segmentation-3.0 revision
`e66f3d3b9eb0873085418a7b813d3b369bf160bb` (MIT, Copyright 2023 CNRS);
community-1 revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
(CC-BY-4.0). Both are gated tier 2. Community attribution: pyannote,
Plaquet & Bredin (segmentation), Wang et al. (WeSpeaker embedding),
Landini et al. (VBx), and BUT Speech@FIT (published PLDA),
https://huggingface.co/pyannote/speaker-diarization-community-1,
https://creativecommons.org/licenses/by/4.0/. Unmodified publisher assets;
no endorsement implied. Keep the attribution with local results. Neither
publisher nor derived weights belong in the repository or an output package.

`pipeline_probe.py` records the community pipeline's documented progress-hook artifacts for one recording
(`tools/scripts/community-long-probe.sh` wraps it for a full meeting plus a short excerpt on the same device).
`../pyannote-gpu` pins the same releases with CUDA wheels of the same Torch version, for reference runs on long
recordings only; set `ONESPEECH_ORACLE_DEVICE=cuda`. On a 180 s excerpt the GPU run's turns were identical to the CPU run's,
embeddings differed by at most 3e-6 and the active-frame count differed by one of about 300,000 segmentation values. `centroid_scan.py` compares the
recorded embeddings with the published speaker centroids under different overlap-free-frame thresholds.
