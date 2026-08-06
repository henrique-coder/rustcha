# Rustcha

`rustcha` recognizes CAPTCHA text from encoded images. The Python API uses a
Rust extension and bundled ONNX models, so inference does not require a network
connection.

## Install

```bash
uv add rustcha
```

The package supports CPython 3.11 and newer. Building from source requires Rust
1.88 or newer.

## Recognize text

Pass encoded image bytes, a path string, or a `pathlib.Path`:

```python
from rustcha import Rustcha

recognizer = Rustcha()
result = recognizer.recognize(image_bytes)

print(result.text)
print(result.confidence)
```

`allowed_characters` constrains decoding to a known alphabet:

```python
digits = recognizer.recognize(image_bytes, allowed_characters="0123456789")
```

Confidence is the geometric mean of the selected character probabilities. It
is `None` when the model returns no characters.

## Detect character regions

The detection model loads on first use. Set `preload_detection=True` if you
prefer to pay that cost when constructing the recognizer.

```python
recognizer = Rustcha(preload_detection=True)
result = recognizer.detect(image_bytes)

for box in result.boxes:
    print(box.x_min, box.y_min, box.x_max, box.y_max, box.confidence)
```

`include_positions=True` pairs recognized characters with detections from left
to right:

```python
result = recognizer.recognize(image_bytes, include_positions=True)

for item in result.character_positions:
    print(item.character, item.box)
```

OCR and detection are separate model passes. Their result counts can differ;
an unmatched character has `box=None`. Call `detect()` when you need every
detection.

## Batches and asyncio

One recognizer reuses one OCR session. `batch_size` limits how many encoded
images enter one native call; the model still evaluates images one at a time.

```python
results = recognizer.batch_recognize(images, batch_size=10)
```

`AsyncRustcha` moves work to a worker thread. Calls on the same instance are
serialized so they can reuse the same native sessions.

```python
from rustcha import AsyncRustcha

recognizer = AsyncRustcha()
result = await recognizer.recognize(image_bytes)
```

## Input limits

Encoded input is limited to 32 MiB. Decoded width and height are each limited
to 8,192 pixels, and image decoders receive a 128 MiB allocation budget. These
limits keep malformed or unexpectedly large inputs from consuming unbounded
memory. They are not a substitute for request-size limits and timeouts in a
service that accepts public uploads.

## Models and license

The Python API, Rust implementation, packaging, tests, and automation in this
repository are original work by Henrique Moreira and are distributed under the
project's MIT license. `rustcha` was inspired by
[`sml2h3/ddddocr`](https://github.com/sml2h3/ddddocr), but has its own
implementation and public API.

The bundled OCR and detection ONNX models come from `ddddocr` and are
distributed under its MIT license. The OCR character map is derived from that
project's `CHARSET_BETA`, so it is attributed with the models. Exact source
details and checksums are recorded in
[THIRD_PARTY_NOTICES.md](https://github.com/henrique-coder/rustcha/blob/prod/THIRD_PARTY_NOTICES.md),
and the upstream license is included in
[licenses/ddddocr-LICENSE](https://github.com/henrique-coder/rustcha/blob/prod/licenses/ddddocr-LICENSE).

OCR output is probabilistic. Do not use it as an authentication or
authorization decision without independent validation.

## Development

```bash
uv sync --all-groups
uv run maturin develop
just check
```

`just check` runs Python lint and type checks, Rust formatting and Clippy,
Rust unit tests, lockfile checks, and Python integration tests.

Every push to `prod` checks the versions in `pyproject.toml`, `Cargo.toml`, and
`uv.lock`. When they match and are newer than the latest `v*` tag, GitHub
Actions builds CPython 3.11+ ABI3 wheels for Linux, macOS, and Windows, publishes
them to PyPI with `uv`, then creates the tag and GitHub release. Release notes
list the commits since the previous tag in Keep a Changelog-style `Added`,
`Changed`, `Fixed`, and `Removed` sections.

The repository must contain a `PYPI_API_TOKEN` Actions secret. A failed release
can be resumed with the `Publish Release` workflow's manual trigger; identical
files already present on PyPI are checked before upload.
