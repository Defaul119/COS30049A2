"""Tests for tools/package.py: which committed files go into the submission zip, and the SHA256SUMS.txt format
(plan §11, Appendix D).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package", ROOT / "tools" / "package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageTests(unittest.TestCase):
    def test_working_notes_and_course_material_stay_out(self) -> None:
        for path in ("CLAUDE.md", "context.txt", "docs/archive/HANDOFF.md", "docs/brief/assignment2_description.pdf"):
            self.assertFalse(package.included(path), path)

    def test_code_data_and_cards_go_in(self) -> None:
        for path in ("README.md", "explorer/cli.py", "data/processed/corpus/ds2.parquet", "DATA_CARD.md", "LICENSES.md",
                     "docs/AI_USE_LOG.md"):
            self.assertTrue(package.included(path), path)

    def test_a_prefix_is_not_a_folder_match(self) -> None:
        self.assertTrue(package.included("docs/briefing.md"))

    def test_sums_are_sorted_in_sha256sum_format(self) -> None:
        text = package.sha256_lines({"b.txt": "2" * 64, "a.txt": "1" * 64})
        self.assertEqual(text, f"{'1' * 64}  a.txt\n{'2' * 64}  b.txt\n")


if __name__ == "__main__":
    unittest.main()
