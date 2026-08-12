# Copyright (c) 2026 Henrique Moreira

from dataclasses import FrozenInstanceError
import math
import unittest

from rustcha import BoundingBox, CharacterPosition, RecognitionResult


class ResultTests(unittest.TestCase):
    def test_results_are_immutable(self) -> None:
        result = RecognitionResult(text="abc", characters=("a", "b", "c"))

        with self.assertRaises(FrozenInstanceError):
            result.text = "changed"  # ty: ignore[invalid-assignment]

    def test_bounding_box_validates_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(x_min=2, y_min=0, x_max=1, y_max=1, confidence=0.5)

    def test_confidence_must_be_finite(self) -> None:
        with self.assertRaises(ValueError):
            RecognitionResult(text="", confidence=math.nan)

    def test_character_cannot_be_empty(self) -> None:
        with self.assertRaises(ValueError):
            CharacterPosition(character="")


if __name__ == "__main__":
    unittest.main()
