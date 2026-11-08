# Speech recognition for .NET.

Offline publisher files for **iic/speech_campplus_sv_en_voxceleb_16k**. Reference this package alongside
RecursiveAI.OneSpeech. Restore copies pinned, digest-addressed files into
`onespeech-models/` beside the application; publish preserves that directory.
Keep it with the executable, including NativeAOT and single-file applications.
OneSpeech verifies all bytes against its catalog, then converts into the per-user
cache on first use. No network is needed for this model. Other pipeline roles
(aligner, diarizer, recognizer) need their own cached files or packages.

The package contains upstream weights, not a model trained or fine-tuned by us.
CAM++ English: ModelScope iic/speech_campplus_sv_en_voxceleb_16k,
revision 032b8131a7ad812f87061955ca974c99060c5a03, Apache-2.0.
CAM++ — Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng and Qian Chen.
Published by 3D-Speaker/ModelScope iic under Apache-2.0.
Publisher-stated training data: VoxCeleb2, 5,994 speakers.

Full weight licences: LICENSE-APACHE and THIRD-PARTY-NOTICES.md in this package.
The library is MIT OR Apache-2.0; that does not relicense publisher weights.
Unaffiliated with other products or companies named OneSpeech.

Publisher: https://modelscope.cn/models/iic/speech_campplus_sv_en_voxceleb_16k
`campplus_voxceleb.bin` SHA-256:
`5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de`.
