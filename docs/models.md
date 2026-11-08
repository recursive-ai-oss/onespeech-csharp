# Model files and pre-fetch

`ModelCatalog.Default` reads a versioned embedded data catalog without network access. `ModelStore.PullAsync(id)` resolves an **explicit** model ID to a `ResolvedModel` containing an absolute directory and `ModelInfo`. The returned directory contains verified publisher files and registered local runtime conversions, including configuration and tokenizer data. Graph execution itself is separate from fetching.

```csharp
var store = new OneSpeech.Catalog.ModelStore(cacheDirectory: "model-cache");
var model = await store.PullAsync("openai/whisper-tiny", cancellationToken: cancellationToken);
// model.Directory: verified publisher files; model.Info: revision, digests and licence provenance.
```

The tool provides `onespeech models list` and `onespeech models pull <id>`. Ctrl+C cancels a pull. List never downloads. No family default is declared until it has been selected from measurements; `GetDefault(family)` refuses when none is declared.

## Cache and trust

Cache location precedence is constructor argument, `ONESPEECH_MODEL_CACHE`, then:

- Windows: LocalApplicationData/OneSpeech/models.
- macOS: `~/Library/Caches/OneSpeech/models`.
- Other platforms: `$XDG_CACHE_HOME/onespeech/models`, falling back to `~/.cache/onespeech/models`.

Use a **trusted per-user directory**, not a shared writable directory. Hosts must not replace cache directories or files with symbolic links. Model subdirectories use a digest of the ID, revision and upstream filenames/digests, so different revisions cannot silently reuse old pins. A `.lock` file opened with exclusive OS sharing serializes pulls across stores and processes. It is intentionally retained after releasing the handle; deleting lock files can split locks between processes. Cancellation also interrupts lock waits. A sharing/I/O failure is retried at 50 ms intervals up to 12,000 times (about ten minutes), then reported as an I/O error.

Each file is downloaded to `.download` under that lock, size-bounded while copying, SHA-256 checked, closed, then atomically renamed to its final name. A cancelled or failed transfer removes its temporary file. An interrupted process can leave `.download`; the next pull restarts it rather than trusting it. Already-complete files survive interruption. There is no byte-range resume.

Present files never cause network requests. On first open **per store instance**, present files are rehashed. The verified marker lives only in memory, keyed by path, length and modification time; a new store/process always rehashes disk bytes. There is no writable disk marker that can authorize unverified weights. A corrupt cache file is removed and produces `ModelIntegrityException`; retry explicitly to fetch it again. The trusted-cache requirement excludes a hostile local process changing bytes while preserving their metadata or holding a file open. This is verification of publisher bytes, not a local filesystem sandbox.

`PullAsync` supports cancellation and `IProgress<ModelDownloadProgress>` with bytes received/expected per file, emitted only for downloads. Errors include `ModelNotFoundException`, `ModelIntegrityException`, `ModelTokenRequiredException`, `ModelNetworkException`, and `InvalidModelCatalogException`. Filesystem permission and storage failures remain ordinary I/O errors. Fetching is asynchronous; there is no synchronous wrapper around network I/O.

## Tiers and sources

Tier 1 is ungated MIT/Apache-2.0/BSD, eligible for defaults. Its transport selects a declared mirror if present, otherwise the publisher. A mirror must be an HTTPS asset beneath `https://github.com/recursive-ai-oss/onespeech-csharp/releases/download/<tag>/<asset>`. All embedded mirror URLs are currently null: **not yet mirrored**. Nothing silently guesses a release URL.

Tier 2 is never a default, has no mirror, and has only pinned publisher URLs. The catalog rejects tier-2 defaults, mirrors, and redistributed converted artifacts before any source is called. Registered tier-2 converters may produce cache-only local artifacts from already verified publisher inputs; those artifacts are never fetched, mirrored or redistributed. Tier 2 may use the declared permissive licences, CC-BY-4.0, or OpenMDW-1.1; unknown, noncommercial and research-only licences are refused. Gate status is independent of licence: any gated entry must be tier 2. Parakeet-TDT-v3 is available by explicit name. Nemotron 3 Diarization is a named tier-2 entry. Its pinned publisher safetensors are read as bounded data and converted under the per-model lock into the user's cache. Local graph/silence artifact digests are registered in code, not as catalog `convertedFiles` URLs. Outputs are hash-checked in a staging directory before installation, never mirrored or packaged; the loader also verifies them. `ModelInfo` retains the OpenMDW agreement and publisher origin.

A gated request requires an explicit `token` argument or `ONESPEECH_MODEL_TOKEN`, even if its bytes are cached or a custom source is used. Gate acceptance is the user's responsibility. Tokens are never logged or persisted. The HTTP source sends authorization only on the original upstream request, not to CDN redirect targets. It follows at most five HTTPS redirects, allowing publisher delivery through signed CDN URLs without adding those URLs to the catalog. Transport failures expose no signed URL or token. Requests time out after 30 minutes; transfers are restartable by calling again.

A host may replace HTTP entirely:

```csharp
var source = new OneSpeech.Catalog.DirectoryModelSource("publisher-files");
var store = new OneSpeech.Catalog.ModelStore(source: source);
```

Directory source layout is `root/<model-id>/<revision>/<filename>`; the root is trusted and must not contain symbolic links. A custom `ModelSource` supplies the pinned publisher bytes. The store still enforces catalog policy, gate authorization, file size and SHA-256. The token-aware virtual overload is additive: older host hooks implementing the tokenless overload still work. A source override does not authorize redistribution of tier-2 files.

## Catalog schema 1

Optional `defaultRoles` is an array of up to 16 unique lower-case ASCII role names
(letters and hyphens, at most 64 characters). It is empty when omitted. Each role
has at most one default across the catalog, independently of `isDefault` family
picks; only ungated tier-1 entries qualify. `GetRoleDefault("transcription")`
selects the file recognizer for `Speech.Transcribe` when no model is specified.
The built-in pick remains `openai/whisper-large-v3-turbo`. Changing that catalog
role assignment changes the pick without changing recognition code. A missing
role default throws `ModelNotFoundException`, never a hidden family fallback.

The envelope is `{ "schemaVersion": 1, "models": [...] }`. Model properties are the camel-case `CatalogEntry` fields: ID, family, source repository, 40-hex revision, files, convertedFiles, tier, licence, exact attribution, gated, isDefault and capabilities. `files` contains the publisher file set. `convertedFiles` is a separate, possibly empty array of local runtime outputs, with SHA-256 and size. Tier-1 output URLs describe OneSpeech release assets. Tier-2 output URLs must exactly match one of the pinned publisher input URLs: they identify provenance, not a downloadable output. `PullAsync` fetches **only publisher files** and produces registered missing converted files locally, then verifies their sizes and hashes before atomic installation. Silero uses publisher ONNX directly. Upstream and converted bytes must never be assigned the same name in one entry.

File metadata is name, size, sha256, upstreamUrl and optional mirrorUrl. Relative names use slash-separated ASCII letters/digits, hyphens, underscores and dots, with an alphanumeric first character per component; Windows device names are also refused; traversal, absolute paths and hidden components are refused. Names are unique ignoring case. Maximum file size is 64 GiB. URLs require HTTPS, no credentials or custom port, and upstream paths pinned to the declared source repository and revision. The publisher forms supported are `<repository>/resolve/<revision>/<file>` and GitHub's `raw.githubusercontent.com/<owner>/<repo>/<revision>/<file>`.

The reader bounds input **before** generated deserialization: 8 MiB UTF-8, depth 32, 64 KiB encoded strings, 500,000 tokens, 100,000 containers, 256 elements per general array, and 64 per file-set array. It accepts at most 256 model entries and 64 files in each file set, rejecting duplicate IDs, duplicate properties, unknown properties, invalid UTF-8, unsupported versions and semantic policy violations. `capabilities`, its `languages` array, and `convertedFiles` are explicit, required fields in schema 1 (use empty arrays where nothing is declared). Empty languages does not mean unrestricted language support. JSON uses `CatalogJsonContext`, generated at compile time, not reflection.

## CAM++ English developer artifact

The CAM++ English entry records the verified publisher checkpoint and the SHA-256/size of its
converted float32 graph. The converted file's content-addressed `models-v1` release URL is a
**reserved, unpublished artifact location**, not a claim that bytes are downloadable. Nothing has
been mirrored or published. Current `ModelStore.PullAsync` fetches publisher `files`, not
`convertedFiles`; run the converter locally until an actual converted asset is available.
No speaker default is selected. See [speaker inference](speakers.md).

ModelScope publisher files use its HTTPS data API, with exactly `Revision=<pinned commit>` and
`FilePath=<catalog filename>` query fields in that order. The catalog checks the API host,
repository path, revision, filename, absence of extra query keys/fragments, and SHA-256/size.
This narrow query exception does not apply to other publishers, mirrors, or converted artifacts.
