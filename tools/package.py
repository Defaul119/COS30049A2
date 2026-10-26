"""Builds the submission zip and SHA256SUMS.txt from a git tag, so the zip holds exactly what was committed.

Serves plan §11 (the zip "unpacks into the tree above", and "SHA256SUMS.txt verifies every file") and Appendix D
(`python tools/package.py --tag v1.0`). Files come from `git archive <tag>`, never from the working folder, so
nothing uncommitted (the DS2 salt, Kaggle tokens, _context/) can reach the zip. Claude's working notes and the
unit's own course material are left out (EXCLUDE). The language-model weights are not in git: --with-models adds
the folders the Assistant uses from models/, hashed like every other file. SHA256SUMS.txt goes into the zip and
beside it, in `sha256sum -c` format, and the finished zip is re-read and checked against it.

Usage:  python tools/package.py --tag v1.0 [--with-models] [--out dist]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "HCMC_COS30049.1_4_Assignment2_Code"
# Not part of the submission: Claude's instructions and running notes (AI use is declared separately), the hand-off
# note between machines, and the unit's brief (course material, not ours to redistribute).
EXCLUDE = ("CLAUDE.md", "context.txt", "docs/HANDOFF.md", "docs/brief/")
# The writers' weights, over GitHub's file limit (README section 8): Qwen 3B round 2's adapter and the 152M model.
MODEL_DIRS = ("models/qwen3b_adapter_v2", "models/assistant")


def included(path: str) -> bool:
    """Whether a committed file belongs in the zip."""
    return not any(path == e or (e.endswith("/") and path.startswith(e)) for e in EXCLUDE)


def tag_files(tag: str) -> dict[str, bytes]:
    """Every file committed at the tag, as git archive writes it (path -> bytes)."""
    tar_bytes = subprocess.run(["git", "archive", "--format=tar", tag], cwd=ROOT, capture_output=True,
                               check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def model_files() -> dict[str, Path]:
    """The writers' weight files from the working folder (path in the zip -> file on disk)."""
    out = {}
    for folder in MODEL_DIRS:
        files = sorted(p for p in (ROOT / folder).glob("*") if p.is_file())
        if not files:
            raise SystemExit(f"{folder} is empty or missing: install the writer first (README section 8)")
        out.update({p.relative_to(ROOT).as_posix(): p for p in files})
    return out


def sha256_lines(hashes: dict[str, str]) -> str:
    """SHA256SUMS.txt: one `<hash>  <path>` line per file, sorted by path (the format `sha256sum -c` reads)."""
    return "".join(f"{hashes[p]}  {p}\n" for p in sorted(hashes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the git tag (or commit) to package, e.g. v1.0")
    parser.add_argument("--with-models", action="store_true", help=f"also add {', '.join(MODEL_DIRS)}")
    parser.add_argument("--out", default="dist", help="output folder (default dist/)")
    args = parser.parse_args()
    files = {p: b for p, b in tag_files(args.tag).items() if included(p)}
    extra = model_files() if args.with_models else {}
    hashes = {p: hashlib.sha256(b).hexdigest() for p, b in files.items()}
    for p, disk in extra.items():
        digest = hashlib.sha256()
        with open(disk, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        hashes[p] = digest.hexdigest()
    sums = sha256_lines(hashes)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in sorted(files):
            z.writestr(f"{NAME}/{p}", files[p])
        for p, disk in sorted(extra.items()):
            z.write(disk, f"{NAME}/{p}")
        z.writestr(f"{NAME}/SHA256SUMS.txt", sums)
    (out_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8", newline="\n")
    # Re-read the finished zip: every file must match its line in SHA256SUMS.txt.
    with zipfile.ZipFile(zip_path) as z:
        bad = [p for p in hashes if hashlib.sha256(z.read(f"{NAME}/{p}")).hexdigest() != hashes[p]]
    if bad:
        raise SystemExit(f"{len(bad)} files in the zip do not match SHA256SUMS.txt, e.g. {bad[0]}")
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(hashes)} files, tag {args.tag}); "
          f"every file matches SHA256SUMS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
