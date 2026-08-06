# Copyright (c) 2026 Henrique Moreira

import asyncio
import random
import struct
import tempfile
import unittest
import zlib

from captcha.image import ImageCaptcha

from rustcha import AsyncRustcha, DetectionResult, RecognitionResult, Rustcha


def captcha_bytes(text: str = "1234") -> bytes:
    random.seed(0)
    return ImageCaptcha(width=160, height=64).generate(text).read()


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
        cls.image = captcha_bytes()

    def test_recognize_bytes_with_character_filter(self) -> None:
        result = self.recognizer.recognize(self.image, allowed_characters="0123456789")

        self.assertIsInstance(result, RecognitionResult)
        self.assertTrue(result.text)
        self.assertTrue(set(result.text) <= set("0123456789"))
        self.assertEqual(result.characters, tuple(result.text))

    def test_recognize_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image_file.write(self.image)
            image_file.flush()
            result = self.recognizer.recognize(image_file.name)

        self.assertIsInstance(result, RecognitionResult)

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


class AsyncRustchaTests(unittest.TestCase):
    def test_recognize(self) -> None:
        async def run() -> RecognitionResult:
            recognizer = AsyncRustcha()
            return await recognizer.recognize(captcha_bytes())

        self.assertIsInstance(asyncio.run(run()), RecognitionResult)


if __name__ == "__main__":
    unittest.main()
