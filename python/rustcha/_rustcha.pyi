# Copyright (c) 2026 Henrique Moreira

from pathlib import Path

__version__: str

class Rustcha:
    def __init__(self) -> None: ...
    def recognize_detailed(
        self,
        image: bytes | str | Path,
        allowed_characters: str | None = None,
    ) -> tuple[str, float | None]: ...
    def batch_recognize_detailed(
        self,
        images: list[bytes | str | Path],
        allowed_characters: str | None = None,
    ) -> list[tuple[str, float | None]]: ...
    def load_detector(self) -> None: ...
    def detect(self, image: bytes | str | Path) -> list[tuple[int, int, int, int, float]]: ...
    def recognize_with_positions(
        self,
        image: bytes | str | Path,
        allowed_characters: str | None = None,
    ) -> tuple[str, float | None, list[tuple[int, int, int, int, float]]]: ...
    def batch_recognize_with_positions(
        self,
        images: list[bytes | str | Path],
        allowed_characters: str | None = None,
    ) -> list[tuple[str, float | None, list[tuple[int, int, int, int, float]]]]: ...
