# Rustcha

`rustcha` recognizes CAPTCHA text from encoded images. The Python API uses a
Rust extension and bundled ONNX models, so inference does not require a network
connection.

## Install

```bash
uv add rustcha
```

The package supports CPython 3.11 through 3.15. Building from source requires
Rust 1.88 or newer.

Pre-built wheels are available for Linux x86_64 and ARM64, Windows x86_64, and
macOS ARM64. Other platforms can build from source when a compatible
ONNX Runtime is available.

## Recognize text

Pass encoded image bytes, a path string, or a `pathlib.Path`:

```python
from rustcha import Rustcha

recognizer = Rustcha()
result = recognizer.recognize(image_bytes, calculate_confidence=True)

print(result.text)
print(result.confidence)
```

`allowed_characters` constrains decoding to a known alphabet:

```python
digits = recognizer.recognize(image_bytes, allowed_characters="0123456789")
```

Confidence calculation is optional because it requires an additional pass over
the model output. Pass `calculate_confidence=True` when needed. Confidence is
the geometric mean of the selected character probabilities and is `None` when
disabled or when the model returns no characters.

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
images enter one native call. The model evaluates images one at a time while a
native pipeline decodes and resizes the next image concurrently.

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

The Rust implementation and Python API were developed by Henrique Moreira and
are distributed under the project's MIT license. `rustcha` was inspired by
[`sml2h3/ddddocr`](https://github.com/sml2h3/ddddocr), but the implementation
and public API are original.

The bundled OCR and detection ONNX models come from `ddddocr` and are
distributed under its MIT license. The OCR character map is derived from that
project's `CHARSET_BETA`, so it is attributed with the models. Exact source
details and checksums are recorded in
[THIRD_PARTY_NOTICES.md](https://github.com/henrique-coder/rustcha/blob/prod/THIRD_PARTY_NOTICES.md),
and the upstream license is included in
[licenses/ddddocr-LICENSE](https://github.com/henrique-coder/rustcha/blob/prod/licenses/ddddocr-LICENSE).

OCR output is probabilistic. Do not use it as an authentication or
authorization decision without independent validation.
