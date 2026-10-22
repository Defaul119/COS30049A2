"""Tests for the DS3/DS4 download step (plan §6.3-6.4): unpacking, checksums and the manual fallback.

A local zip served through a file:// URL stands in for UCI and Kaggle, so no network is needed.
Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from explorer import external
from explorer.settings import SETTINGS


def spec(url: str, files: list[str]) -> dict:
    return {"id": "DSX", "name": "Test set", "page": "https://example.org", "url": url, "doi": None,
            "citation": "Test (2026)", "licence": "CC BY 4.0", "licenceCheckedOn": "2026-10-22", "expectedFiles": files}


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.external_dir, self.processed_dir = root / "external", root / "processed"
        self.source = root / "source.zip"
        with zipfile.ZipFile(self.source, "w") as zf:
            zf.writestr("nested/data.csv", "a,b\n1,2\n3,4\n")
            zf.writestr("nested/readme.txt", "not needed")

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self, key: str):
        return external.fetch_dataset(key, self.external_dir, self.processed_dir)

    def test_download_unpacks_expected_files_and_records_checksums(self):
        with mock.patch.dict(SETTINGS["external"], {"fake": spec(self.source.as_uri(), ["data.csv"])}):
            record = self.fetch("fake")
            extracted = self.external_dir / "fake" / "data.csv"
            self.assertEqual(record["files"][0]["sha256"], hashlib.sha256(extracted.read_bytes()).hexdigest())
            self.assertEqual(record["files"][0]["lines"], 3)
            self.assertFalse((self.external_dir / "fake" / "readme.txt").exists())
            saved = json.loads((self.processed_dir / "fake" / external.RECORD_FILE).read_text(encoding="utf-8"))
            self.assertEqual(saved, record)

            # Files on disk are not fetched again, and an unchanged record keeps its retrieval time.
            self.source.unlink()
            again = self.fetch("fake")
            self.assertEqual(again["retrieved_utc"], record["retrieved_utc"])

    def test_archive_without_an_expected_file_is_rejected_and_removed(self):
        with mock.patch.dict(SETTINGS["external"], {"fake": spec(self.source.as_uri(), ["missing.csv"])}):
            with self.assertRaises(ValueError):
                self.fetch("fake")
            self.assertFalse((self.external_dir / "fake" / "fake.zip").exists())

    def test_elliptic_without_a_token_asks_for_the_manual_steps(self):
        empty = Path(self.tmp.name) / "no-kaggle"
        empty.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
        env["KAGGLE_CONFIG_DIR"] = str(empty)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(external.ManualDownloadNeeded):
                self.fetch("elliptic")

    def test_a_zip_placed_by_hand_is_used(self):
        folder = self.external_dir / "elliptic"
        folder.mkdir(parents=True)
        names = SETTINGS["external"]["elliptic"]["expectedFiles"]
        with zipfile.ZipFile(folder / "elliptic-data-set.zip", "w") as zf:
            for name in names:
                zf.writestr(f"elliptic_bitcoin_dataset/{name}", "1,2\n")
        record = self.fetch("elliptic")
        self.assertEqual([f["file"] for f in record["files"]], names)
        self.assertEqual(record["archive"]["file"], "elliptic-data-set.zip")


if __name__ == "__main__":
    unittest.main()
