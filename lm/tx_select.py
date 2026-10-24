"""Choose TxGPT's size from the 9 sweep runs and export the chosen weights for scoring.

Serves plan §9.1 ("the size with the lowest mean validation loss wins") and §10.1 (validation
loss by size and seed). Reads every runs/txgpt-<size>-s<seed>/model_best.pt (the lowest validation
loss each run reached), writes one row per run and one per size, picks the size with the lowest
mean validation loss and, within it, the seed with the lowest, and exports that run's weights in
fp16 with the token definitions they were trained on, so lm.tx_score runs from models/ alone.

Usage:
    python -m lm.tx_select

Writes reports/tables/txgpt_runs.csv, reports/tables/txgpt_sizes.csv, models/txgpt.pt and
models/txgpt_meta.json (a copy of data/txgpt/meta.json).
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import torch

RUN_NAME = re.compile(r"txgpt-(?P<size>[a-z]+)-s(?P<seed>\d+)$")
SIZE_ORDER = {"s": 0, "m": 1, "l": 2}


def run_table(runs_dir: Path) -> pd.DataFrame:
    """One row per finished sweep run: size, seed, parameters, best validation loss and when."""
    rows = []
    for folder in sorted(runs_dir.glob("txgpt-*-s*")):
        match = RUN_NAME.match(folder.name)
        best = folder / "model_best.pt"
        if not match or not best.exists():
            continue
        ckpt = torch.load(best, map_location="cpu", weights_only=True)
        last = folder / "ckpt_last.pt"
        state = torch.load(last, map_location="cpu", weights_only=True) if last.exists() else {}
        rows.append({
            "size": match["size"], "seed": int(match["seed"]),
            # The output layer shares the embedding matrix, so count each tensor once.
            "parameters": sum(t.numel() for k, t in ckpt["model"].items() if k != "lm_head.weight"),
            "best_val_loss": float(ckpt["val_loss"]), "best_step": int(ckpt["step"]),
            "last_step": int(state.get("step", ckpt["step"])), "stopped_early": bool(state.get("stopped_early", False)),
            "finished": (folder / "model.pt").exists(), "path": best.as_posix(),
        })
    if not rows:
        raise SystemExit(f"no finished TxGPT runs under {runs_dir} (expected txgpt-<size>-s<seed>/model_best.pt)")
    table = pd.DataFrame(rows)
    return table.sort_values(["size", "seed"], key=lambda c: c.map(SIZE_ORDER) if c.name == "size" else c,
                             ignore_index=True)


def size_table(runs: pd.DataFrame) -> pd.DataFrame:
    sizes = runs.groupby("size", sort=False).agg(
        parameters=("parameters", "first"), runs=("seed", "size"), mean_val_loss=("best_val_loss", "mean"),
        std_val_loss=("best_val_loss", "std"), min_val_loss=("best_val_loss", "min")).reset_index()
    sizes["chosen"] = sizes["mean_val_loss"] == sizes["mean_val_loss"].min()
    return sizes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--meta", default="data/txgpt/meta.json")
    p.add_argument("--tables", default="reports/tables")
    p.add_argument("--models", default="models")
    args = p.parse_args()
    tables, models = Path(args.tables), Path(args.models)
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    runs = run_table(Path(args.runs))
    unfinished = runs[~runs["finished"]]
    if len(unfinished):
        print(f"[tx_select] warning: {len(unfinished)} run(s) not finished yet: {unfinished['path'].tolist()}")
    sizes = size_table(runs)
    chosen_size = sizes.loc[sizes["chosen"], "size"].iloc[0]
    chosen = runs[runs["size"] == chosen_size].sort_values("best_val_loss").iloc[0]
    runs["chosen"] = runs["path"] == chosen["path"]
    runs.to_csv(tables / "txgpt_runs.csv", index=False)
    sizes.to_csv(tables / "txgpt_sizes.csv", index=False)
    print(sizes.to_string(index=False))

    ckpt = torch.load(chosen["path"], map_location="cpu", weights_only=True)
    # fp16 halves the file; lm.tx_score loads it into a float32 model (load_state_dict casts).
    export = {"model": {k: t.half() for k, t in ckpt["model"].items()}, "model_config": ckpt["model_config"],
              "step": ckpt["step"], "val_loss": ckpt["val_loss"], "kind": ckpt.get("kind"),
              "source": chosen["path"], "size": chosen_size, "seed": int(chosen["seed"])}
    torch.save(export, models / "txgpt.pt")
    shutil.copyfile(args.meta, models / "txgpt_meta.json")
    print(f"[tx_select] chose size {chosen_size.upper()} (lowest mean validation loss), seed {chosen['seed']} "
          f"(validation loss {chosen['best_val_loss']:.4f}) -> {models / 'txgpt.pt'}")


if __name__ == "__main__":
    main()
