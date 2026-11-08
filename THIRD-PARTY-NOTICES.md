# Third-party notices

## Model weights

Core embeds the Silero VAD v6.2.3 ONNX release model. No other model weights are included.

## Data

The LibriSpeech clip identified below is the only third-party audio fixture; ID-only LibriSpeech evaluation lists are also included, with attribution below. Other feature fixtures use deterministic synthetic signals.

## Libraries

Package dependencies retain their own licences and notices.

## Silero VAD v6.2.3

The embedded silero_vad.onnx release model is distributed under the following licence.

Upstream licence: https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/LICENSE

Model SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Downloaded Whisper checkpoints

Whisper weights are not embedded. First use downloads the pinned publisher files and
converts them locally. The five catalog entries retain the original OpenAI MIT notice,
including "Copyright (c) 2022 OpenAI", in the embedded catalog attribution. The pinned
Hugging Face cards declare Apache-2.0 for tiny, base, small and large-v3, and MIT for
large-v3-turbo. The Apache-2.0 text is included in LICENSE-APACHE. Source repositories,
revisions and file digests are recorded in the catalog and model provenance.

## Optional speaker checkpoints

The following publisher checkpoints are not included in this package. English CAM++ is
downloaded and converted on first use of the default diarizer; the other entries are
pinned by the development fetch tool. The model
cards declare Apache-2.0; the licence text is included in LICENSE-APACHE. These are published
pretrained weights, not models trained or fine-tuned by this project. The training-data statements
below are the publishers' descriptions, not an independent audit or a licence grant for that data.

- SpeechBrain ECAPA-TDNN, `speechbrain/spkrec-ecapa-voxceleb` revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, `embedding_model.ckpt`,
  SHA-256 `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`.
  Source: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb .
  Publisher-stated training data: VoxCeleb1 and VoxCeleb2.
- CAM++ English, `iic/speech_campplus_sv_en_voxceleb_16k` revision
  `032b8131a7ad812f87061955ca974c99060c5a03`, `campplus_voxceleb.bin`,
  SHA-256 `5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de`.
  Source: https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k .
  Publisher-stated training data: VoxCeleb2, 5,994 speakers.
- CAM++ Chinese common, `iic/speech_campplus_sv_zh-cn_16k-common` revision
  `013c18884dffb82846cf472dd06dd4a08d97e2cd`, `campplus_cn_common.bin`,
  SHA-256 `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8`.
  Source: https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common .
  Publisher-stated training data: a large Chinese speaker dataset with over 100,000 speakers;
  the card does not identify the corpus by name.
- ERes2NetV2 Chinese common, `iic/speech_eres2netv2_sv_zh-cn_16k-common` revision
  `1cf80d41fb3435bd3d8df185b5c423333b2db42a`, `pretrained_eres2netv2.ckpt`,
  SHA-256 `0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c`.
  Source: https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common .
  Publisher-stated training data: a large Chinese speaker dataset with approximately 200,000
  speakers; the card does not identify the corpus by name.

## LibriSpeech evaluation identifiers

`eval/manifests/librispeech-speaker-trials.tsv` and
`eval/manifests/librispeech-speaker-parity50.txt` contain utterance identifiers derived from
LibriSpeech test-clean/test-other (OpenSLR resource 12), by Vassil Panayotov, Guoguo Chen,
Daniel Povey and Sanjeev Khudanpur (2015).

Source: https://www.openslr.org/12/ . Dataset licence: Creative Commons Attribution 4.0
International, https://creativecommons.org/licenses/by/4.0/ .

OneSpeech selected and paired these identifiers for evaluation; no original audio or transcript
text is included in these files. The selection, pairing method and seed are documented in
`docs/speakers.md`. This attribution does not imply endorsement by the dataset authors.

## Reference observations

The tone feature fixtures were produced from deterministic synthetic audio by black-box calls to openai-whisper 20250625 (MIT). They contain only numerical outputs, not model weights, source code or third-party audio. The oracle harness also supports transformers 5.22.2 (Apache-2.0). Neither oracle is a shipped dependency.

## LibriSpeech fixture

`testdata/1089-134686-0001.flac` is from the LibriSpeech ASR corpus (2015), Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur. Test-clean utterance 1089-134686-0001 is distributed unmodified under Creative Commons Attribution 4.0: <https://creativecommons.org/licenses/by/4.0/>. Source: <https://www.openslr.org/12>. The archive path and SHA-256 are recorded in `testdata/manifest.json`.

### ECAPA reference-only auxiliary files

The pinned `speechbrain/spkrec-ecapa-voxceleb` reference also initializes `classifier.ckpt`
(5,534,328 bytes, SHA-256 `fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535`)
and `mean_var_norm_emb.ckpt` (1,921 bytes, SHA-256
`cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33`). They are under the same
publisher-declared Apache-2.0 licence at revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.
These files are used only by the separate evaluation oracle; no classifier or global-embedding
normalization weights are included in OneSpeech's converted embedding graph or package.

## Qwen3-ASR and Qwen3-ForcedAligner

Qwen Team, Qwen3-ASR and Qwen3-ForcedAligner. The published model cards declare
Apache License 2.0. The full licence is provided in LICENSE-APACHE. Model data
and converted weights are downloaded separately, not embedded in this package.
Pinned checkpoint revisions and file digests are in the model catalog; model
provenance and attribution are retained in the returned ModelInfo.

## Omnilingual ASR checkpoint and character vocabulary

Copyright 2025 Meta Platforms, Inc. and affiliates.

The optional `facebook/omniASR-CTC-300M` checkpoint and vocabulary are published
by Meta AI / the Omnilingual ASR Team under Apache License 2.0, reproduced in
`LICENSE-APACHE`. Source:
https://huggingface.co/facebook/omniASR-CTC-300M/tree/8e35f0cc28fa6099e0c14d56db85ce0423baa691

OneSpeech converts the published tensors to its runtime format without training
or fine-tuning. Weights are downloaded separately; they are not embedded in the
core package. Retain the licence and these attribution notices when
redistributing the model or converted weights.

## Moonshine English speech recognition weights

The tiny, base, streaming-tiny, streaming-small and streaming-medium checkpoints
from moonshine-ai are MIT-licensed. Their pinned model cards and source links
are in the model catalog. No Community License checkpoints are included.
Publisher licence: https://github.com/moonshine-ai/moonshine/blob/main/LICENSE

MIT License

Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
