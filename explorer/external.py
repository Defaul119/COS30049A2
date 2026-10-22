"""DS3 (Elliptic) and DS4 (BitcoinHeist): download, unpack, and record checksums and licences.

Serves plan §6.3 and §6.4 (docs/plan/sections/s05-data.tex) and DR-01: every external file is
recorded with its source, licence, retrieval time, size and SHA-256, so a marker who downloads it
again can confirm they have the same bytes.

BitcoinHeist is a direct UCI download. Elliptic comes from Kaggle, which needs the user's API token
(%USERPROFILE%\\.kaggle\\kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY). Without a token the step
prints the manual steps instead; a zip or the CSVs placed by hand in data/external/elliptic/ are
then picked up by the next run.

Writes:
    data/external/<dataset>/             the archive and its extracted files (git-ignored: 0.2-0.7 GB)
    data/processed/<dataset>/download.json   the record above (committed)
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .esplora import USER_AGENT
from .settings import PROJECT_ROOT, SETTINGS

log = logging.getLogger(__name__)

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RECORD_FILE = "download.json"
CHUNK_BYTES = 1 << 20  # read and hash 1 MiB at a time, so 0.7 GB files never sit in memory
PROGRESS_EVERY_BYTES = 50 << 20


class ManualDownloadNeeded(RuntimeError):
    """The dataset cannot be fetched automatically; the message says what to do instead."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(CHUNK_BYTES), b""))


def kaggle_auth() -> str | None:
    """The HTTP Basic credentials for the Kaggle API, from the environment or kaggle.json; None if absent."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
        token = config_dir / "kaggle.json"
        if not token.exists():
            return None
        creds = json.loads(token.read_text(encoding="utf-8"))
        user, key = creds.get("username"), creds.get("key")
    return base64.b64encode(f"{user}:{key}".encode()).decode() if user and key else None


def download(url: str, dest: Path, timeout: float, basic_auth: str | None = None) -> None:
    """Stream `url` to `dest` through a .part file, so an interrupted download is never mistaken for a whole one.

    Raises urllib.error.URLError (or HTTPError) if the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if basic_auth:
        # Unredirected: Kaggle answers with a redirect to a signed storage URL, which must not get the credentials.
        request.add_unredirected_header("Authorization", f"Basic {basic_auth}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    received, next_report = 0, PROGRESS_EVERY_BYTES
    with urllib.request.urlopen(request, timeout=timeout) as resp, part.open("wb") as out:
        while chunk := resp.read(CHUNK_BYTES):
            out.write(chunk)
            received += len(chunk)
            if received >= next_report:
                log.info("  %s: %d MB so far", dest.name, received >> 20)
                next_report += PROGRESS_EVERY_BYTES
    part.replace(dest)
    log.info("Downloaded %s (%d MB)", dest.name, received >> 20)


def extract_expected(archive: Path, folder: Path, expected: list[str]) -> None:
    """Extract only the expected files, by base name, flat into `folder`.

    Taking base names avoids trusting paths stored inside a downloaded archive, and copes with the
    Kaggle zip keeping its CSVs in a sub-folder.
    """
    with zipfile.ZipFile(archive) as zf:
        members = {Path(info.filename).name: info for info in zf.infolist() if not info.is_dir()}
        missing = [name for name in expected if name not in members]
        if missing:
            raise ValueError(f"{archive.name} lacks {missing}; it holds {sorted(members)}")
        for name in expected:
            target = folder / name
            if not target.exists():
                with zf.open(members[name]) as src, target.open("wb") as out:
                    while chunk := src.read(CHUNK_BYTES):
                        out.write(chunk)
                log.info("Extracted %s", name)


def fetch_dataset(key: str, external_dir: Path = EXTERNAL_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, Any]:
    """Make sure the dataset's expected files are on disk, then write and return its download record.

    `key` is "bitcoinheist" or "elliptic" (settings.json -> external). Files already present are not
    downloaded again. Raises ManualDownloadNeeded when Elliptic has no token and no local copy.
    """
    spec = SETTINGS["external"][key]
    folder = external_dir / key
    folder.mkdir(parents=True, exist_ok=True)
    expected = spec["expectedFiles"]
    archive = folder / f"{key}.zip"

    if not all((folder / name).exists() for name in expected):
        if not archive.exists():
            hand_placed = sorted(p for p in folder.glob("*.zip") if p != archive)
            if hand_placed:
                archive = hand_placed[0]
                log.info("Using the archive placed by hand: %s", archive.name)
            else:
                auth = kaggle_auth() if key == "elliptic" else None
                if key == "elliptic" and auth is None:
                    raise ManualDownloadNeeded(
                        f"No Kaggle API token found. Either create one (kaggle.com -> Settings -> API -> Create New "
                        f"Token) and save it as {Path.home() / '.kaggle' / 'kaggle.json'}, or download the zip from "
                        f"{spec['page']} and put it in {folder}. Then run this step again."
                    )
                log.info("Downloading %s from %s", spec["name"], spec["url"])
                download(spec["url"], archive, SETTINGS["external"]["downloadTimeoutSeconds"], auth)
        try:
            extract_expected(archive, folder, expected)
        except (zipfile.BadZipFile, ValueError):
            if archive == folder / f"{key}.zip":
                archive.unlink()  # our own download is broken (e.g. an error page); fetch it afresh next run
            raise

    record_path = processed_dir / key / RECORD_FILE
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    files = [
        {"file": name, "bytes": (folder / name).stat().st_size, "sha256": sha256_file(folder / name),
         "lines": count_lines(folder / name)}
        for name in expected
    ]
    if previous.get("files") and previous["files"] != files:
        # The source changed since the last record: results built on the old files no longer match.
        log.warning("%s differs from the recorded download; record the change in CHANGELOG.md", key)
    archive_info = (
        {"file": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
        if archive.exists() else previous.get("archive")
    )
    record = {
        "dataset": spec["id"],
        "name": spec["name"],
        "page": spec["page"],
        "url": spec["url"],
        "doi": spec["doi"],
        "citation": spec["citation"],
        "licence": spec["licence"],
        "licence_checked_on": spec["licenceCheckedOn"],
        "retrieved_utc": previous.get("retrieved_utc") if previous.get("files") == files
        else datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": archive_info,
        "files": files,
        "location": f"data/external/{key}/ (git-ignored; rebuilt with python -m explorer.cli external --{key})",
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for f in files:
        log.info("%s: %d bytes, %d lines, sha256 %s", f["file"], f["bytes"], f["lines"], f["sha256"])
    if spec["licence"] is None:
        log.warning("No licence recorded for %s: read it on %s and set it in settings.json -> external",
                    key, spec["page"])
    log.info("Wrote %s", record_path)
    return record


def run_external(keys: list[str]) -> int:
    """Fetch each dataset in `keys`. Returns 0 if all are on disk and recorded, 1 otherwise."""
    code = 0
    for key in keys:
        try:
            fetch_dataset(key)
        except ManualDownloadNeeded as err:
            log.warning("%s: %s", key, err)
            code = 1
        except (urllib.error.URLError, zipfile.BadZipFile, ValueError) as err:
            log.error("%s failed: %s. Run the same command again to retry.", key, err)
            code = 1
    return code
