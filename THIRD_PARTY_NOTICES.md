# Third-party notices

## ddddocr models and character map

The following files come from the `sml2h3/ddddocr` project at commit
`c40f56f95412e10bcb9bd0bd24411e92f896d238`:

| File | Upstream source | SHA-256 |
| --- | --- | --- |
| `python/rustcha/models/ocr/common.onnx` | `ddddocr/common.onnx` | `33b5cd351ee94e73a6bf8fa18c415ed8b819b3ffd342e267c30d8ad8334e34e8` |
| `python/rustcha/models/detection/common_det.onnx` | `ddddocr/common_det.onnx` | `6faa8ea85a8c1a634e5050c4a138fca10f30194e0d7abbe9ade1fcd423af6ed6` |

`python/rustcha/models/ocr/characters.json` contains `CHARSET_BETA` from
`ddddocr/charsets.py`, wrapped with local format metadata. Its SHA-256 is
`33b4e0ee1a6a29cb6a2226b0a0c0ef674a8ac34bbcdcac50970cab4a03f3ca4b`.

Upstream repository: <https://github.com/sml2h3/ddddocr>

These assets are distributed under the upstream MIT license in
`licenses/ddddocr-LICENSE`.
