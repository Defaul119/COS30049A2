"""Tests for the committed Kaggle notebooks (kaggle/<name>/, written by tools/kaggle_notebooks.py): every one is valid
Python pinned to a full commit, and its metadata is a template the push copies can fill in.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = sorted(p for p in (ROOT / "kaggle").iterdir() if p.is_dir())


def code_of(notebook: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


class KaggleNotebookTests(unittest.TestCase):
    def test_every_notebook_compiles_and_is_pinned_to_a_commit(self) -> None:
        self.assertTrue(FOLDERS)
        for folder in FOLDERS:
            with self.subTest(folder=folder.name):
                notebook = folder / f"{folder.name}.ipynb"
                code = code_of(notebook)
                compile(code, str(notebook), "exec")
                self.assertRegex(code, r'COMMIT = "[0-9a-f]{40}"')  # a full hash already on GitHub

    def test_metadata_is_a_template_matching_its_notebook(self) -> None:
        for folder in FOLDERS:
            with self.subTest(folder=folder.name):
                meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
                self.assertTrue(meta["id"].startswith("KAGGLE_USERNAME/cos30049a2-"))
                self.assertEqual(meta["code_file"], f"{folder.name}.ipynb")
                self.assertTrue(meta["is_private"])
                for source in meta["kernel_sources"] + meta["dataset_sources"]:
                    self.assertRegex(source, r"^(KAGGLE_USERNAME|ellipticco)/[a-z0-9-]+$")
                # GPU notebooks stop at once without a GPU; CPU notebooks never ask for one
                needs_gpu = 'shutil.which("nvidia-smi") is None' in code_of(folder / f"{folder.name}.ipynb")
                self.assertEqual(meta["enable_gpu"], needs_gpu)


if __name__ == "__main__":
    unittest.main()
