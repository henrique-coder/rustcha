# Copyright (c) 2026 Henrique Moreira

import asyncio
from pathlib import Path
import struct
import unittest
import zlib

from rustcha import AsyncRustcha, DetectionResult, RecognitionResult, Rustcha


ROOT = Path(__file__).parents[1]
SAMPLE_PATH = ROOT / "samples" / "captcha.png"
SAMPLE_TEXT = "3n3d"


def sample_bytes() -> bytes:
    return SAMPLE_PATH.read_bytes()


def png_bytes(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\0" + bytes(width * 3) for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


class RustchaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recognizer = Rustcha()
        cls.image = sample_bytes()

    def test_recognize_bytes_with_character_filter(self) -> None:
        result = self.recognizer.recognize(self.image, allowed_characters="0123456789")

        self.assertIsInstance(result, RecognitionResult)
        self.assertTrue(result.text)
        self.assertTrue(set(result.text) <= set("0123456789"))
        self.assertEqual(result.characters, tuple(result.text))
        self.assertIsNone(result.confidence)

    def test_confidence_is_opt_in(self) -> None:
        result = self.recognizer.recognize(self.image, calculate_confidence=True)

        self.assertIsNotNone(result.confidence)

    def test_recognize_path(self) -> None:
        result = self.recognizer.recognize(SAMPLE_PATH)

        self.assertIsInstance(result, RecognitionResult)
        self.assertEqual(result.text, SAMPLE_TEXT)

    def test_detect(self) -> None:
        result = self.recognizer.detect(self.image)

        self.assertIsInstance(result, DetectionResult)
        self.assertTrue(result.boxes)
        self.assertTrue(all(box.x_min <= box.x_max and box.y_min <= box.y_max for box in result.boxes))

    def test_invalid_image_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.recognizer.recognize(b"not an image")

    def test_oversized_encoded_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 MiB"):
            self.recognizer.recognize(bytes(32 * 1024 * 1024 + 1))

    def test_oversized_image_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.recognizer.recognize(png_bytes(8193, 1))

    def test_batch_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self.recognizer.batch_recognize([self.image], batch_size=0)

    def test_batch_matches_single_recognition(self) -> None:
        single = self.recognizer.recognize(self.image)
        batch = self.recognizer.batch_recognize([self.image, self.image])

        self.assertEqual([result.text for result in batch], [single.text, single.text])


class AsyncRustchaTests(unittest.TestCase):
    def test_recognize(self) -> None:
        async def run() -> RecognitionResult:
            recognizer = AsyncRustcha()
            return await recognizer.recognize(sample_bytes())

        self.assertIsInstance(asyncio.run(run()), RecognitionResult)


if __name__ == "__main__":
    unittest.main()
