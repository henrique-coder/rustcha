# Benchmarks

`rustcha` can be compared with the original Python implementation, `ddddocr`,
using the same encoded image bytes. The benchmark measures the complete public
Python call (image decoding, preprocessing, ONNX inference, decoding, and the
Python/native boundary), not an artificially isolated kernel.

The comparison initializes `ddddocr` with `beta=True`. Both implementations
therefore use the same `common.onnx` model; the checked-in and upstream model
files have the same SHA-256 checksum.

Both text-recognition paths run without probability/confidence calculation.
This measures the common text-only capability and avoids charging either side
for optional output that the other side did not request.

## Reproduce locally

The benchmark dependencies are isolated in the `benchmark` dependency group.
Build the optimized extension and sync that group:

```bash
uv sync --group benchmark
uv run maturin develop --release
uv run python benchmarks/benchmark.py \
  --json target/benchmarks/latest.json \
  --report target/benchmarks/latest.md
```

To measure only this project, which is useful on machines where `ddddocr` is
not available:

```bash
uv run maturin develop --release
uv run python benchmarks/benchmark.py --backend rustcha
```

The default corpus contains the checked-in sample plus CAPTCHA images generated
once with `captcha`. It uses volumes of 1, 10, 100, and 500 images. The script
keeps the resulting encoded bytes in memory and passes those exact same bytes
to both backends; it never regenerates an image for the second implementation.
The output includes a SHA-256 fingerprint of the complete corpus. `single_api`
makes one public call per image; `batch_api` exercises
`Rustcha.batch_recognize` with chunks of 10.
The reference has no equivalent batch API. Its `single_api` row is the fair
baseline; `rustcha`'s `batch_api` rows are reported separately as an additional
capability, not as an equivalent operation.

The output reports cold start, median elapsed time, and images per second. Run
several times on an otherwise idle machine and compare medians on the same
hardware. CPU model, OS, Python version, ONNX Runtime version, thread count,
thermal state, and background load all affect the result. Do not compare a
local laptop result with a GitHub-hosted runner as if they were the same host.

Output agreement is only a diagnostic. The generated corpus has no ground
truth labels, and equal output is not proof of accuracy. For an accuracy claim,
keep a labelled, representative corpus and report exact-match accuracy and
failure cases separately from speed.

## GitHub Actions

The manually triggered [`benchmarks.yml`](../.github/workflows/benchmarks.yml)
runs a matrix over Ubuntu, Windows, and macOS, with Python 3.11 through 3.15,
and installs the project with
`uv sync --frozen --group benchmark`. That group contains both `captcha` and
`ddddocr`; no dependency is installed manually in the workflow. The generated
JSON and Markdown reports are uploaded as artifacts and the key values are copied to the GitHub
Actions job summary. The artifact is retained for 30 days.

At the time of writing, `ddddocr`'s `onnxruntime` dependency does not publish a
Python 3.15 wheel. Python 3.15 therefore runs the `rustcha`-only benchmark on
each operating system, while Python 3.11 through 3.14 run the fair two-library
comparison.

The final repository receives only the benchmark script, dependency-group
declaration, lockfile, documentation, and workflow. The generated images stay
in the runner's memory and the JSON stays in the workflow artifact; neither is
committed automatically. Download an artifact from the Actions run when you
need to archive or compare a result. This is intended for trend tracking, not
as a merge gate: hosted runner hardware can change. Pin a self-hosted runner
or a dedicated benchmark host when regression thresholds matter.

```bash
gh workflow run benchmarks.yml
```
