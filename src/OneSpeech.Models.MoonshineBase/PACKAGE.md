# Speech recognition for .NET.

Offline publisher files for **moonshine-ai/moonshine-base**. Reference this package alongside
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

Publisher: https://huggingface.co/moonshine-ai/moonshine-base
Revision: `7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf`.
`model.safetensors` SHA-256:
`e020c79d0a979a7ec099f718ff1cd2f19e92aead230d69654bca5975a8e1b862`.
