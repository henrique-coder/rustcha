default:
    @just --list

sync:
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

lock-check:
    uv lock --check

test: develop
    uv run python -m unittest discover -s tests

check: lint typecheck rust-check lock-check test

develop:
    uv run maturin develop --release

build:
    uv run maturin build --release --out target/wheels

clean:
    cargo clean
