# Copyright (c) 2026 Henrique Moreira

"""Python interface for CAPTCHA text recognition."""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from ._rustcha import Rustcha as _NativeRustcha
from ._rustcha import __version__
from .results import BoundingBox, CharacterPosition, DetectionResult, RecognitionResult


ImageSource: TypeAlias = bytes | str | Path
"""Supported source for one encoded image."""


class Rustcha:
    """Recognize CAPTCHA text and locate character-like regions."""

    def __init__(self, *, preload_detection: bool = False) -> None:
        """Load the OCR model and create a recognizer.

        Args:
            preload_detection: Load the bundled detection model during
                construction. Leave this as ``False`` to defer its one-time
                startup cost until detection or positions are requested.
        """
        self._native = _NativeRustcha()
        self._detection_is_loaded = False

        if preload_detection:
            self._ensure_detection_loaded()

    def recognize(
        self,
        image: ImageSource,
        allowed_characters: str | None = None,
        *,
        include_positions: bool = False,
    ) -> RecognitionResult:
        """Recognize text from one CAPTCHA image.

        Args:
            image: Encoded image bytes, a filesystem path string, or a
                :class:`pathlib.Path`.
            allowed_characters: Restrict returned characters to this string.
                The OCR model still evaluates its complete character map.
            include_positions: Include each recognized character's detected
                bounding box. This loads the detector once when needed.

        Returns:
            The recognized text, confidence, characters, and optional positions.

        Raises:
            ValueError: If the image cannot be read or decoded.
            RuntimeError: If ONNX model loading or inference fails.
        """
        if include_positions:
            self._ensure_detection_loaded()
            text, confidence, raw_boxes = self._native.recognize_with_positions(image, allowed_characters)
            return self._recognition_result_with_positions(text, confidence, raw_boxes)

        text, confidence = self._native.recognize_detailed(image, allowed_characters)
        return self._recognition_result(text, confidence)

    def batch_recognize(
        self,
        images: Sequence[ImageSource],
        allowed_characters: str | None = None,
        *,
        batch_size: int = 10,
        include_positions: bool = False,
    ) -> list[RecognitionResult]:
        """Recognize CAPTCHA images in input order.

        Args:
            images: Ordered image bytes or filesystem paths.
            allowed_characters: Restrict returned characters to this string.
            batch_size: Maximum images held by one native call. The default of
                ``10`` is a memory-conscious starting point; it can be any
                positive integer.
            include_positions: Include character positions for every result.
                This loads the detector once when needed.

        Returns:
            One result per input image, in exactly the input order.

        Raises:
            ValueError: If ``batch_size`` is less than one or an image cannot
                be read or decoded.
            RuntimeError: If ONNX model loading or inference fails.

        The bundled model evaluates one image at a time. Chunks cap how many
        encoded images cross into Rust in one call. Rust releases the GIL while
        it decodes and evaluates each chunk.
        """
        if batch_size < 1:
            message = "batch_size must be at least 1"
            raise ValueError(message)

        if include_positions:
            self._ensure_detection_loaded()

        results: list[RecognitionResult] = []
        for start in range(0, len(images), batch_size):
            chunk = list(images[start : start + batch_size])
            if include_positions:
                raw_results = self._native.batch_recognize_with_positions(chunk, allowed_characters)
                results.extend(
                    self._recognition_result_with_positions(text, confidence, boxes)
                    for text, confidence, boxes in raw_results
                )
            else:
                raw_results = self._native.batch_recognize_detailed(chunk, allowed_characters)
                results.extend(self._recognition_result(text, confidence) for text, confidence in raw_results)
        return results

    def detect(self, image: ImageSource) -> DetectionResult:
        """Detect character-like regions without performing OCR.

        Args:
            image: Encoded image bytes, a filesystem path string, or a
                :class:`pathlib.Path`.

        Returns:
            Detected boxes and their detector confidence, in detector score
            order. This is useful for custom cropping, visualization, or a
            recognition workflow outside :meth:`recognize`.

        Raises:
            ValueError: If the image cannot be read or decoded.
            RuntimeError: If detection model loading or inference fails.
        """
        self._ensure_detection_loaded()
        raw_boxes = self._native.detect(image)
        return DetectionResult(boxes=self._boxes(raw_boxes))

    def _ensure_detection_loaded(self) -> None:
        if not self._detection_is_loaded:
            self._native.load_detector()
            self._detection_is_loaded = True

    @staticmethod
    def _recognition_result(text: str, confidence: float | None) -> RecognitionResult:
        return RecognitionResult(text=text, confidence=confidence, characters=tuple(text))

    @classmethod
    def _recognition_result_with_positions(
        cls,
        text: str,
        confidence: float | None,
        raw_boxes: Sequence[tuple[int, int, int, int, float]],
    ) -> RecognitionResult:
        boxes = tuple(sorted(cls._boxes(raw_boxes), key=lambda box: box.x_min))
        return RecognitionResult(
            text=text,
            confidence=confidence,
            characters=tuple(text),
            character_positions=tuple(
                CharacterPosition(character=character, box=boxes[index] if index < len(boxes) else None)
                for index, character in enumerate(text)
            ),
        )

    @staticmethod
    def _boxes(raw_boxes: Sequence[tuple[int, int, int, int, float]]) -> tuple[BoundingBox, ...]:
        return tuple(
            BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=confidence,
            )
            for x_min, y_min, x_max, y_max, confidence in raw_boxes
        )


class AsyncRustcha:
    """Run one shared recognizer in a worker thread, one operation at a time."""

    def __init__(self, *, preload_detection: bool = False) -> None:
        """Create an asynchronous recognizer.

        Args:
            preload_detection: Load the detection model immediately instead of
                waiting for the first detection or position request.
        """
        self._recognizer = Rustcha(preload_detection=preload_detection)
        self._operation_lock = asyncio.Lock()

    async def recognize(
        self,
        image: ImageSource,
        allowed_characters: str | None = None,
        *,
        include_positions: bool = False,
    ) -> RecognitionResult:
        """Recognize one image without blocking the running event loop.

        Args:
            image: Encoded image bytes, a filesystem path string, or a path.
            allowed_characters: Restrict returned characters to this string.
            include_positions: Include detected boxes for each character.

        Returns:
            The recognition result.
        """
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._recognizer.recognize,
                image,
                allowed_characters,
                include_positions=include_positions,
            )

    async def batch_recognize(
        self,
        images: Sequence[ImageSource],
        allowed_characters: str | None = None,
        *,
        batch_size: int = 10,
        include_positions: bool = False,
    ) -> list[RecognitionResult]:
        """Recognize an ordered batch without blocking the running event loop.

        Args:
            images: Ordered image bytes or filesystem paths.
            allowed_characters: Restrict returned characters to this string.
            batch_size: Maximum images in one native operation; defaults to 10.
            include_positions: Include character boxes for every image.

        Returns:
            One result per input image in input order.
        """
        async with self._operation_lock:
            return await asyncio.to_thread(
                self._recognizer.batch_recognize,
                images,
                allowed_characters,
                batch_size=batch_size,
                include_positions=include_positions,
            )

    async def detect(self, image: ImageSource) -> DetectionResult:
        """Detect image regions without blocking the running event loop.

        Args:
            image: Encoded image bytes, a filesystem path string, or a path.

        Returns:
            The detected bounding boxes.
        """
        async with self._operation_lock:
            return await asyncio.to_thread(self._recognizer.detect, image)


__all__ = [
    "AsyncRustcha",
    "BoundingBox",
    "CharacterPosition",
    "DetectionResult",
    "ImageSource",
    "RecognitionResult",
    "Rustcha",
    "__version__",
]
