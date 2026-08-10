# Copyright (c) 2026 Henrique Moreira

"""Compare rustcha with ddddocr on the same deterministic image corpus.

The script intentionally measures the public Python APIs, including image
decoding and the Python-to-native boundary. It is not a model-only benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch


if TYPE_CHECKING:
    from collections.abc import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUMES = (1, 10, 100, 500)
TEXTS = (
    "A7k2",
    "mP4x",
    "8qZ1",
    "b3Nw",
    "K9rt",
    "v2Hd",
    "C6ys",
    "pL8e",
    "R4aM",
    "t7Qf",
    "2Wnb",
    "x5Jk",
    "H8cz",
    "n6Vp",
    "D1rg",
    "s9LX",
)


@dataclass(frozen=True)
class Measurement:
    backend: str
    volume: int
    operation: str
    median_seconds: float
    images_per_second: float
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("both", "rustcha", "ddddocr"), default="both")
    parser.add_argument("--count", type=int, default=500, help="Largest generated corpus size (default: 500).")
    parser.add_argument("--iterations", type=int, default=3, help="Measured repetitions per case (default: 3).")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup calls per backend (default: 2).")
    parser.add_argument("--json", type=Path, help="Also write machine-readable results to this path.")
    parser.add_argument("--report", type=Path, help="Also write a readable Markdown report to this path.")
    args = parser.parse_args()
    if args.count < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("count and iterations must be positive; warmup cannot be negative")
    return args


def make_corpus(count: int) -> list[bytes]:
    """Generate deterministic CAPTCHA-like PNGs, with the checked-in sample first."""
    from captcha.image import ImageCaptcha

    corpus: list[bytes] = []
    sample = ROOT / "samples" / "captcha.png"
    if sample.exists():
        corpus.append(sample.read_bytes())
    generator = ImageCaptcha(width=160, height=60)
    randomizer = random.Random(1337)
    with (
        patch("captcha.image.secrets.choice", randomizer.choice),
        patch("captcha.image.secrets.randbelow", randomizer.randrange),
        patch("captcha.image.secrets.randbits", randomizer.getrandbits),
    ):
        for index in range(max(0, count - len(corpus))):
            stream = generator.generate(TEXTS[index % len(TEXTS)])
            corpus.append(stream.getvalue())
    return corpus[:count]


def corpus_fingerprint(corpus: Sequence[bytes]) -> str:
    """Return a stable fingerprint for the exact encoded images being measured."""
    digest = hashlib.sha256()
    for image in corpus:
        digest.update(len(image).to_bytes(8, "big"))
        digest.update(image)
    return digest.hexdigest()


def load_backend(backend: str) -> Any:
    if backend == "rustcha":
        from rustcha import Rustcha

        return Rustcha()
    if backend == "ddddocr":
        try:
            import ddddocr
        except ImportError as error:
            message = "ddddocr is not installed; use `uv sync --group benchmark` or run with --backend rustcha"
            raise SystemExit(message) from error
        return ddddocr.DdddOcr(beta=True, show_ad=False)
    raise ValueError(f"unsupported backend: {backend}")


def recognize(backend: str, recognizer: Any, image: bytes) -> str:
    if backend == "rustcha":
        return recognizer.recognize(image, calculate_confidence=False).text
    result = recognizer.classification(image)
    return result if isinstance(result, str) else str(result)


def run_case(
    backend: str,
    recognizer: Any,
    images: Sequence[bytes],
    operation: str,
    iterations: int,
    warmup: int,
) -> tuple[Measurement, list[str]]:
    for image in images[:warmup]:
        recognize(backend, recognizer, image)

    durations: list[float] = []
    outputs: list[str] = []
    for _ in range(iterations):
        start = time.perf_counter()
        if operation == "batch_api" and backend == "rustcha":
            outputs = [
                result.text
                for result in recognizer.batch_recognize(
                    images,
                    batch_size=10,
                    calculate_confidence=False,
                )
            ]
        else:
            outputs = [recognize(backend, recognizer, image) for image in images]
        durations.append(time.perf_counter() - start)
    elapsed = statistics.median(durations)
    return Measurement(backend, len(images), operation, elapsed, len(images) / elapsed, iterations), outputs


def print_results(measurements: Sequence[Measurement], agreement: float | None) -> None:
    print(f"Python {platform.python_version()} | {platform.platform()} | iterations={measurements[0].samples}")
    print("\nbackend  volume  operation   median(s)  images/s")
    print("-------  ------  ----------  ---------  --------")
    for item in measurements:
        print(
            f"{item.backend:7}  {item.volume:6}  {item.operation:10}  {item.median_seconds:9.4f}  {item.images_per_second:8.2f}"
        )
    if agreement is not None:
        print(f"\nOutput agreement (same text): {agreement:.1%}")
        print("Agreement is diagnostic, not an accuracy score: this corpus has no ground-truth labels.")


def write_report(
    path: Path,
    measurements: Sequence[Measurement],
    agreement: float | None,
    corpus: Sequence[bytes],
    fingerprint: str,
    cold_starts: dict[str, float],
) -> None:
    """Write a readable report alongside the technical JSON result."""
    single = [item for item in measurements if item.operation == "single_api"]
    volumes = sorted({item.volume for item in single})
    lines = [
        "# Benchmark report",
        "",
        f"Environment: Python {platform.python_version()} on {platform.platform()}",
        f"Corpus: {len(corpus)} encoded images",
        f"Corpus SHA-256: `{fingerprint}`",
        "Confidence calculation: disabled for both implementations",
        "",
        "## Intuitive summary",
        "",
        "The equivalent comparison is `single_api`, which both libraries expose.",
    ]
    for volume in volumes:
        rows = [item for item in single if item.volume == volume]
        if len(rows) == 2:
            fastest = max(rows, key=lambda item: item.images_per_second)
            lines.append(
                f"- {volume} image(s): **{fastest.backend}** had the highest throughput "
                f"({fastest.images_per_second:.2f} images/s)."
            )
    if agreement is not None:
        lines.extend(
            [
                "",
                f"Output agreement between the equivalent calls: **{agreement:.1%}**.",
                "This is not an accuracy score because the generated corpus has no labels.",
            ]
        )
    rust_batch = [item for item in measurements if item.backend == "rustcha" and item.operation == "batch_api"]
    if rust_batch:
        lines.extend(
            [
                "",
                "`rustcha` also exposes `batch_api`; `ddddocr` has no corresponding batch method.",
                "Its throughput is reported separately and must not be presented as an equivalent API comparison.",
            ]
        )
    lines.extend(["", "## Technical report", "", "### Cold start", "", "| Backend | Seconds |", "| --- | ---: |"])
    lines.extend(f"| {backend} | {seconds:.4f} |" for backend, seconds in cold_starts.items())
    lines.extend(
        [
            "",
            "### Measurements",
            "",
            "| Backend | Volume | Operation | Median (s) | Images/s |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {item.backend} | {item.volume} | {item.operation} | {item.median_seconds:.4f} | {item.images_per_second:.2f} |"
        for item in measurements
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    names = ("rustcha", "ddddocr") if args.backend == "both" else (args.backend,)
    corpus = make_corpus(args.count)
    fingerprint = corpus_fingerprint(corpus)
    print(f"Corpus: {len(corpus)} encoded images | SHA-256: {fingerprint}")
    measurements: list[Measurement] = []
    outputs: dict[str, list[str]] = {}
    cold_starts: dict[str, float] = {}

    for backend in names:
        start = time.perf_counter()
        recognizer = load_backend(backend)
        cold_start = time.perf_counter() - start
        cold_starts[backend] = cold_start
        print(f"{backend} cold start: {cold_start:.4f}s")
        for volume in DEFAULT_VOLUMES:
            if volume > len(corpus):
                continue
            operations = ("single_api", "batch_api") if backend == "rustcha" else ("single_api",)
            for operation in operations:
                measurement, result = run_case(
                    backend, recognizer, corpus[:volume], operation, args.iterations, args.warmup
                )
                measurements.append(measurement)
                if volume == min(DEFAULT_VOLUMES[-1], len(corpus)) and operation == "single_api":
                    outputs[backend] = result

    agreement = None
    if len(outputs) == 2:
        agreement = sum(
            left == right for left, right in zip(outputs["rustcha"], outputs["ddddocr"], strict=True)
        ) / len(outputs["rustcha"])
    print_results(measurements, agreement)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "corpus_count": len(corpus),
                    "corpus_sha256": fingerprint,
                    "calculate_confidence": False,
                    "cold_start_seconds": cold_starts,
                    "measurements": [asdict(item) for item in measurements],
                    "agreement": agreement,
                },
                indent=2,
            )
            + "\n"
        )
    if args.report:
        write_report(args.report, measurements, agreement, corpus, fingerprint, cold_starts)


if __name__ == "__main__":
    main()
