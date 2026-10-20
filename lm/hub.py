"""Download datasets from the Hugging Face Hub as Parquet files.

Whole files are downloaded instead of streamed: a dropped connection then only
costs the current file (downloads resume), and data preparation can restart
exactly where it stopped.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

# Every external dataset the language models use, with its licence for the
# report's data section and the dataset manifest.
SOURCES = {
    "fineweb-edu": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "prefix": "sample/10BT/",
        "column": "text",
        "license": "ODC-By 1.0",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    },
    "smol-smoltalk": {
        "repo_id": "HuggingFaceTB/smol-smoltalk",
        "prefix": "data/train",
        "column": "messages",
        "license": "Apache 2.0",
        "url": "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk",
    },
}


def list_files(source: str) -> list[str]:
    s = SOURCES[source]
    files = HfApi().list_repo_files(s["repo_id"], repo_type="dataset")
    return sorted(f for f in files if f.startswith(s["prefix"]) and f.endswith(".parquet"))


def download(source: str, filename: str, local_dir: Path) -> Path:
    s = SOURCES[source]
    path = hf_hub_download(s["repo_id"], filename, repo_type="dataset", local_dir=str(local_dir))
    return Path(path)


def iter_column(path: Path, column: str, start_row: int = 0, batch_rows: int = 1024):
    """Yield (row_number, value) for one column of a Parquet file, from start_row on."""
    row = 0
    for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=batch_rows, columns=[column]):
        n = batch.num_rows
        if row + n <= start_row:
            row += n
            continue
        values = batch.column(0).to_pylist()
        for i in range(max(0, start_row - row), n):
            yield row + i, values[i]
        row += n
