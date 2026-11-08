# Speaker-change locator probe

`locator.py` runs the documented public `speaker-diarization` pipeline for
`iic/speech_campplus-transformer_scl_zh-cn_16k-common` at revision
`c4e62f7b842dd4231b85adcbd7d5b587ed91a5c1`, using ModelScope 1.40.2 on CPU
in the environment pinned by this directory's `uv.lock`. It does not inspect
model objects or installed package files. Network connections are disabled.

```sh
python locator.py LOCATOR_CACHE AMI_DATA_ROOT SCRATCH_AUDIO OUTPUT_JSON
```

The locator cache must contain the two publisher weight files and
`configuration.json` at the SHA-256 pins checked by the script. The data root
must contain `prepared/ES2011a.wav` and `prepared/ES2011b.wav` from the AMI
v1.6.2 preparation scripts. Only development audio is used: eight-second
excerpts starting at 0, 60 and 120 seconds in each meeting. Input and weight
hashes, oracle version, precision and complete returned data are recorded.
Audio, embeddings and weights are not fixtures and must not be committed.
The scratch directory contains derived audio and may be removed after the run.

## Observed output and limits

On these six excerpts the public pipeline returned a dictionary containing
only `text`, for example `The change point is at 2.31s.` It did not return
frame scores or an encoder output. This is evidence about the public API,
not evidence of native graph parity or accurate boundaries. The card limits
the model to two-person conversations known to contain a change point;
there is no no-change or overlap-detection claim.

The publisher configuration specifies 16 kHz, 80 filterbank dimensions,
`frame_size=512` and `anchor_size=192`. The card describes frame features,
optional global speaker embeddings and a transformer backend, but not the
complete inference graph. Chen et al., *3D-Speaker-Toolkit*, arXiv
2403.19971v3 §§II.A–II.B describe CAM++ and clustering/multimodal modules,
not this acoustic locator's detailed transformer architecture. Native
conversion needs a complete architecture specification or published model
graph; this probe does not establish one. No locator graph or locator-backed
diarization is shipped by this tooling.

## Attribution and terms

The locator publisher card declares Apache-2.0 and states that training used
a large-scale synthetic Chinese two-person audio dataset. The probe uses the
released weights without training or alteration. The AMI Meeting Corpus is
CC-BY-4.0, University of Edinburgh and the AMI Consortium,
<https://groups.inf.ed.ac.uk/ami/corpus/>. Excerpts are modifications for
local evaluation only; retain corpus notices with the data, and do not
redistribute them from this tool. Model and dataset acquisition pins are in
`diarization-bundle.tsv` and the existing AMI data fetch/preparation tooling.
