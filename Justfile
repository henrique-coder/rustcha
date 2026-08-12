default:
    @just --list

update:
    uv sync --upgrade --all-groups --all-extras

format:
    uv run --group dev ruff format python tests
    uv run --group dev ruff check --fix python tests
    cargo fmt

lint:
    uv run --group dev ruff format --check python tests
    uv run --group dev ruff check python tests

typecheck:
    uv run --group dev ty check python

rust-check:
    cargo fmt --check
    cargo clippy --locked --all-targets --all-features -- -D warnings
    cargo test --locked

develop:
    uv run maturin develop --release

test: develop
    uv run python -m unittest discover -s tests

check: lint typecheck rust-check test

build:
    uv run maturin build --release --out target/wheels
