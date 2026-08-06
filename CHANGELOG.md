# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-06

### Added

- **Recognition API:** Added synchronous, asynchronous, and batch CAPTCHA text recognition.
- **Character detection:** Added optional character-region detection and position-aware recognition.
- **Offline models:** Bundled the OCR and detection ONNX models for inference without a network connection.
- **Typed results:** Added immutable Python result objects and native type information.
- **Release automation:** Added ABI3 wheel builds for Linux, macOS, and Windows with automatic PyPI and GitHub publication.

### Changed

- **Package metadata:** Synchronized the shared metadata in `pyproject.toml` and `Cargo.toml`, including description, authors, license, URLs, README, and keywords.
- **Model provenance:** Documented the source, checksums, and license of the bundled models and character map.

### Fixed

- **Image resource limits:** Bounded encoded input size, decoded dimensions, decoder allocation, and batch processing.
- **Detector grid:** Corrected the stride boundary used when decoding the detector output.
- **Model decoding:** Added shape, score, character-map, and finite-value validation before returning results.

### Removed

- **Unused dependencies and API:** Removed Pydantic, Rich, the arbitrary detector-path loader, and redundant raw native recognition methods.
