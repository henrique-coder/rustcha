# Copyright (c) 2026 Henrique Moreira

"""Result types returned by rustcha."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A detected image region in pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float

    def __post_init__(self) -> None:
        """Reject invalid coordinates and confidence values."""
        if min(self.x_min, self.y_min, self.x_max, self.y_max) < 0:
            message = "bounding-box coordinates cannot be negative"
            raise ValueError(message)
        if self.x_min > self.x_max or self.y_min > self.y_max:
            message = "bounding-box minimums cannot exceed their maximums"
            raise ValueError(message)
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class CharacterPosition:
    """A recognized character and its detected region, when available."""

    character: str
    box: BoundingBox | None = None

    def __post_init__(self) -> None:
        """Require at least one character."""
        if not self.character:
            message = "character cannot be empty"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Text and optional confidence returned by the OCR model."""

    text: str
    confidence: float | None = None
    characters: tuple[str, ...] = ()
    character_positions: tuple[CharacterPosition, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid confidence values."""
        if self.confidence is not None:
            _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Bounding boxes returned by the detection model."""

    boxes: tuple[BoundingBox, ...] = ()


def _validate_confidence(value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        message = "confidence must be finite and between 0 and 1"
        raise ValueError(message)


__all__ = ["BoundingBox", "CharacterPosition", "DetectionResult", "RecognitionResult"]
