# Speech recognition for .NET.

Offline publisher files for **moonshine-ai/moonshine-tiny**. Reference this package alongside
RecursiveAI.OneSpeech. Restore copies pinned, digest-addressed files into
`onespeech-models/` beside the application; publish preserves that directory.
Keep it with the executable, including NativeAOT and single-file applications.
OneSpeech verifies all bytes against its catalog, then converts into the per-user
cache on first use. No network is needed for this model. Other pipeline roles
(aligner, diarizer, recognizer) need their own cached files or packages.

The package contains upstream weights, not a model trained or fine-tuned by us.
Moonshine English weights: Useful Sensors, Inc. (dba Moonshine AI), MIT.
Publisher-stated training data: tiny/base, 200,000 hours of internet audio and
open datasets; streaming-tiny, roughly 300,000 hours including an additional
100,000 hours of internally prepared speech data. See THIRD-PARTY-NOTICES.md
for the publisher attribution, licence text and model revisions.

Full weight licences: LICENSE-APACHE and THIRD-PARTY-NOTICES.md in this package.
The library is MIT OR Apache-2.0; that does not relicense publisher weights.
Unaffiliated with other products or companies named OneSpeech.

Publisher: https://huggingface.co/moonshine-ai/moonshine-tiny
Revision: `390624ed33d594443aa4aa221f5b9f283b545b5a`.
`model.safetensors` SHA-256:
`867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736`.
