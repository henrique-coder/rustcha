# Copyright (c) 2026 Henrique Moreira

from pathlib import Path
import tomllib
import unittest

import rustcha


ROOT = Path(__file__).parents[1]


class MetadataTests(unittest.TestCase):
    def test_versions_match(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]
        with (ROOT / "Cargo.toml").open("rb") as file:
            package = tomllib.load(file)["package"]
        with (ROOT / "uv.lock").open("rb") as file:
            packages = tomllib.load(file)["package"]
        lock_version = next(package["version"] for package in packages if package["name"] == "rustcha")

        self.assertEqual(project["version"], package["version"])
        self.assertEqual(project["version"], lock_version)
        self.assertEqual(project["version"], rustcha.__version__)

    def test_shared_package_metadata_match(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]
        with (ROOT / "Cargo.toml").open("rb") as file:
            package = tomllib.load(file)["package"]

        python_author = project["authors"][0]
        self.assertEqual(project["name"], package["name"])
        self.assertEqual(project["description"], package["description"])
        self.assertEqual(project["license"], package["license"])
        self.assertEqual(project["readme"], package["readme"])
        self.assertEqual(project["keywords"], package["keywords"])
        self.assertEqual(project["urls"]["Homepage"], package["homepage"])
        self.assertEqual(project["urls"]["Repository"], package["repository"])
        self.assertEqual(
            f"{python_author['name']} <{python_author['email']}>",
            package["authors"][0],
        )


if __name__ == "__main__":
    unittest.main()
