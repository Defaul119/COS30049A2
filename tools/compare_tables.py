"""Compare rebuilt result tables with the committed ones, number by number (README sections 4 and 11; milestone M4).

After `python -m explorer.cli all --offline` and `figures`, every CSV under reports/tables/ is compared with the copy
committed in git (or with the same file in another folder): the same columns and rows, equal text cells, and every
number within tolerance x max(1, |committed value|) (settings.json -> rebuildCheck; README 11: a difference above
0.01 in a recall or a rate is worth reporting). Columns that record when or where a run happened (time, commit,
seconds, the model file used) are not compared.

Usage:
    python tools/compare_tables.py                    # reports/tables/ against git HEAD
    python tools/compare_tables.py --ref <folder>     # against another folder of tables (same relative paths)
    python tools/compare_tables.py --report out.csv   # also write one row per table

Prints one line per table that differs, then a summary; exits 1 if any table differs beyond the tolerance.
Tables that exist on only one side are listed but do not fail the check (a skipped step, e.g. without the DS4
download, leaves its tables as committed).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
CHECK = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))["rebuildCheck"]


def committed(rel: str, ref: Path | None) -> pd.DataFrame | None:
    """The reference copy of reports/tables/<rel>: from git HEAD, or from the folder `ref`; None if it has none."""
    if ref is not None:
        path = ref / rel
        return pd.read_csv(path) if path.exists() else None
    shown = subprocess.run(["git", "show", f"HEAD:reports/tables/{rel}"], cwd=ROOT, capture_output=True)
    return pd.read_csv(io.BytesIO(shown.stdout)) if shown.returncode == 0 else None


def compare(new: pd.DataFrame, old: pd.DataFrame, tolerance: float) -> tuple[str, float]:
    """("same" | "within" | "differs: <why>", the largest scaled numeric difference)."""
    keep = [c for c in old.columns if c not in CHECK["ignoreColumns"]]
    if list(new.columns) != list(old.columns):
        return f"differs: columns {sorted(set(new.columns) ^ set(old.columns)) or 'in another order'}", float("nan")
    if len(new) != len(old):
        return f"differs: {len(new)} rows, committed {len(old)}", float("nan")
    worst, why = 0.0, []
    for c in keep:
        a, b = new[c], old[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
            both_missing = np.isnan(x) & np.isnan(y)
            if (np.isnan(x) != np.isnan(y)).any():
                why.append(f"{c}: missing values differ")
                continue
            scaled = np.abs(x - y) / np.maximum(1.0, np.abs(y))
            scaled[both_missing | (x == y)] = 0.0  # equal infinities give nan above
            column_worst = float(np.nanmax(scaled)) if len(scaled) else 0.0
            worst = max(worst, column_worst)
            if column_worst > tolerance:
                why.append(f"{c}: up to {column_worst:.4g}")
        elif not a.astype(str).equals(b.astype(str)):
            why.append(f"{c}: text differs")
    if why:
        return "differs: " + "; ".join(why[:4]) + (" ..." if len(why) > 4 else ""), worst
    return ("same" if worst == 0 else "within"), worst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, help="a folder of reference tables (default: git HEAD)")
    p.add_argument("--tolerance", type=float, default=CHECK["tolerance"])
    p.add_argument("--report", type=Path, help="write one row per table to this CSV")
    args = p.parse_args()
    rows = []
    for path in sorted(TABLES.rglob("*.csv")):
        rel = path.relative_to(TABLES).as_posix()
        old = committed(rel, args.ref)
        if old is None:
            rows.append({"table": rel, "result": "new (no committed copy)", "worst": float("nan")})
            continue
        result, worst = compare(pd.read_csv(path), old, args.tolerance)
        rows.append({"table": rel, "result": result, "worst": worst})
    table = pd.DataFrame(rows)
    for r in table.itertuples():
        if r.result not in ("same", "within"):
            print(f"{r.table}: {r.result}")
    counts = table["result"].str.split(":").str[0].value_counts().to_dict()
    print(f"{len(table)} tables: {counts} (tolerance {args.tolerance} x max(1, |committed|))")
    if args.report:
        table.to_csv(args.report, index=False)
    return int(table["result"].str.startswith("differs").any())


if __name__ == "__main__":
    sys.exit(main())
