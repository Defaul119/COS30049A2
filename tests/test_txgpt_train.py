"""End-to-end check of TxGPT's data and training options on CPU: prepare_tx, --seed and early stopping.

Builds a tiny synthetic DS2 table, writes token files with lm.prepare_tx, and trains a tiny model with
lm.train. A learning rate of 0 means validation can never improve, so early stopping must stop the
run and leave model.pt, and running the command again must not train further.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def synthetic_ds2(n_addresses: int = 60, n_tx: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_addresses):
        t = 1_700_000_000
        for i in range(n_tx):
            t += int(rng.exponential(20_000))
            received = int(rng.lognormal(12, 1)) if i % 2 == 0 else 0
            rows.append({"addr_id": f"{a:016x}", "txid_hash": f"{a:08x}{i:08x}", "block_height": 800_000 + t // 600,
                         "block_time_utc": t, "received_sats": received, "sent_sats": 0 if received else 5000,
                         "n_inputs": 1 + i % 3, "n_outputs": 2, "fee_sats": 300, "vsize_vb": 141})
    return pd.DataFrame(rows)


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=cwd,
                          env={**os.environ, "PYTHONUTF8": "1"})


class TxGPTTrainingTests(unittest.TestCase):
    def test_prepare_seed_suffix_and_early_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_ds2().to_parquet(tmp / "ds2.parquet", index=False)
            prep = run("-m", "lm.prepare_tx", "--source", str(tmp / "ds2.parquet"), "--out", str(tmp / "tx"), cwd=ROOT)
            self.assertEqual(prep.returncode, 0, prep.stderr)
            meta = json.loads((tmp / "tx" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["vocab_size"], 94)
            self.assertEqual(meta["split"]["train"]["addresses"] + meta["split"]["val"]["addresses"]
                             + meta["split"]["test"]["addresses"], 60)
            config = {"name": "txgpt-test", "data_dir": str(tmp / "tx"), "out_dir": str(tmp / "run"),
                      "model": {"seq_len": 768, "n_layer": 1, "n_head": 2, "d_model": 32, "mlp_hidden": 64, "dropout": 0.0},
                      "epochs": 50, "batch_tokens": 1536, "micro_batch": 2, "lr": 0.0, "min_lr": 0.0, "warmup_steps": 1,
                      "eval_every": 2, "eval_batches": 1, "early_stop_evals": 2, "ckpt_every_minutes": 60}
            (tmp / "cfg.json").write_text(json.dumps(config), encoding="utf-8")
            first = run("-m", "lm.train", "--config", str(tmp / "cfg.json"), "--device", "cpu", "--seed", "2", cwd=ROOT)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            self.assertIn("stopping early", first.stdout)
            self.assertTrue((tmp / "run-s2" / "model.pt").exists())
            self.assertTrue((tmp / "run-s2" / "model_best.pt").exists())
            again = run("-m", "lm.train", "--config", str(tmp / "cfg.json"), "--device", "cpu", "--seed", "2", cwd=ROOT)
            self.assertIn("already finished", again.stdout)


if __name__ == "__main__":
    unittest.main()
