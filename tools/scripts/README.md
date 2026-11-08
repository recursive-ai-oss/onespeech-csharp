# Fetch and install scripts

Scripts take a caller-selected cache outside the checkout. They never upload data or credentials.

- `fetch-librispeech.sh TARGET`: pinned test-clean/test-other archives, CC BY 4.0; verify before extraction.
- `fetch-earnings22.sh TARGET`: media and original NLP transcripts at a pinned publisher commit, CC BY-SA 4.0, evaluation only. Existing files are verified, not silently replaced.
- `install-oracles.sh TARGET`: three isolated, locked CPU reference environments. Requires `uv` and Python 3.13 (uv can fetch Python); installs no system packages.

Requires Bash, curl, sha256sum, sha1sum, tar and standard Unix utilities. URLs, terms and content hashes are kept beside fetched data. The committed `pins/` files contain hashes only, not dataset content. Oracle lockfiles are in `eval/drivers/`. See [evaluation](../../docs/eval.md) for formats, settings and commands.
