default:
    @just --list

update:
    uv sync --upgrade --all-groups --all-extras

format:
    uv run ruff format python
    uv run ruff check --fix python
    cargo fmt

lint:
    uv run ruff check python tests

typecheck:
    uv run ty check python

rust-check:
    cargo fmt --check
    cargo clippy --locked --all-targets --all-features -- -D warnings
    cargo test --locked

test: develop
    uv run python -m unittest discover -s tests

develop:
    uv run maturin develop --release

build:
    uv run maturin build --release --out target/wheels

benchmark:
    uv run maturin develop --release
    uv run benchmarks/benchmark.py --json target/benchmarks/latest.json --report target/benchmarks/latest.md
