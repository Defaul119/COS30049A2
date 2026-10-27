"""Round 2's training options on the CPU (docs/KAGGLE_LM_PLAN.md, C1-C4).

The warm-up/stable/decay schedule, --max-hours (save, stop, resume), validation windows that do not
depend on the micro-batch, --eval-only, two processes training exactly as one (lm.dist, run here with
gloo on the CPU), and prepare_text --continue-from.

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lm.train import lr_at

ROOT = Path(__file__).resolve().parent.parent
VOCAB = 300


def tiny_corpus(folder: Path, seed: int = 0) -> None:
    """Random token files in lm.prepare_text's layout (enough for a few steps of a tiny model)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name, n in (("train.bin", 40_000), ("val.bin", 8_000)):
        rng.integers(0, VOCAB, n, dtype=np.uint16).tofile(folder / name)
    (folder / "meta.json").write_text(json.dumps({"kind": "text-pretraining", "vocab_size": VOCAB,
                                                  "train_tokens": 40_000, "val_tokens": 8_000}), encoding="utf-8")


def write_config(tmp: Path, name: str, **override) -> Path:
    cfg = {"name": name, "data_dir": str(tmp / "data"), "out_dir": str(tmp / name),
           "model": {"seq_len": 64, "n_layer": 1, "n_head": 2, "d_model": 32, "mlp_hidden": 64, "dropout": 0.0},
           "total_tokens": 64 * 8 * 3, "batch_tokens": 64 * 8, "micro_batch": 2, "lr": 1e-3, "min_lr": 1e-4,
           "warmup_steps": 1, "eval_every": 100, "eval_batches": 2, "ckpt_every_minutes": 60, **override}
    path = tmp / f"{name}.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(module: str, *args: str, world: int = 1) -> list[subprocess.CompletedProcess]:
    """python -m module args, as `world` processes with torchrun's environment variables (gloo on the CPU).

    torchrun itself is not used: its rendezvous store needs libuv, which PyTorch's Windows builds lack.
    """
    env = {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": ""}
    if world > 1:
        env.update(WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(free_port()), USE_LIBUV="0")
    # Output goes to files, not pipes: a process blocked on a full pipe would never reach the collective
    # the others wait in, and the run would hang.
    with tempfile.TemporaryDirectory() as logs:
        files = [(open(Path(logs) / f"{r}.out", "w+", encoding="utf-8"), open(Path(logs) / f"{r}.err", "w+", encoding="utf-8"))
                 for r in range(world)]
        procs = [subprocess.Popen([sys.executable, "-m", module, *args], cwd=ROOT, text=True, stdout=out, stderr=err,
                                  env={**env, "RANK": str(r), "LOCAL_RANK": str(r)} if world > 1 else env)
                 for r, (out, err) in enumerate(files)]
        done = []
        for p, (out, err) in zip(procs, files):
            p.wait(timeout=600)
            out.seek(0), err.seek(0)
            done.append(subprocess.CompletedProcess(p.args, p.returncode, out.read(), err.read()))
            out.close(), err.close()
    return done


def weights(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


class ScheduleTests(unittest.TestCase):
    def test_wsd_is_constant_until_the_last_fraction_then_linear(self):
        cfg = {"lr": 1.0, "min_lr": 0.1, "warmup_steps": 10, "schedule": "wsd", "decay_fraction": 0.2}
        self.assertAlmostEqual(lr_at(4, cfg, 100), 0.5)
        self.assertEqual([lr_at(s, cfg, 100) for s in (10, 50, 80)], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(lr_at(90, cfg, 100), 0.55)
        self.assertAlmostEqual(lr_at(100, cfg, 100), 0.1)
        # More tokens before the decay began: the run so far is unchanged, so total_tokens can be raised late.
        self.assertEqual(lr_at(70, cfg, 100), lr_at(70, cfg, 400))

    def test_cosine_stays_the_default(self):
        cfg = {"lr": 1.0, "min_lr": 0.0, "warmup_steps": 0}
        self.assertAlmostEqual(lr_at(50, cfg, 100), 0.5)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        tiny_corpus(self.tmp / "data")

    def test_two_processes_train_exactly_as_one(self):
        one = run("lm.train", "--config", str(write_config(self.tmp, "one")), "--device", "cpu")
        two = run("lm.train", "--config", str(write_config(self.tmp, "two")), "--device", "cpu", world=2)
        for p in one + two:
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        a, b = weights(self.tmp / "one" / "model.pt"), weights(self.tmp / "two" / "model.pt")
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=1e-6)
        self.assertNotIn("[train]", two[1].stdout)  # only rank 0 prints

    def test_max_hours_saves_stops_and_resumes(self):
        cfg = str(write_config(self.tmp, "timed", total_tokens=64 * 8 * 4))
        first = run("lm.train", "--config", cfg, "--device", "cpu", "--max-hours", "1e-9")[0]
        self.assertEqual(first.returncode, 0, first.stderr[-2000:])
        self.assertIn("--max-hours", first.stdout)
        self.assertTrue((self.tmp / "timed" / "ckpt_last.pt").exists())
        self.assertFalse((self.tmp / "timed" / "model.pt").exists())
        rest = run("lm.train", "--config", cfg, "--device", "cpu")[0]
        self.assertIn("resuming from step 1", rest.stdout)
        self.assertEqual(torch.load(self.tmp / "timed" / "model.pt", weights_only=True)["step"], 4)

    def test_eval_windows_do_not_depend_on_the_micro_batch(self):
        losses = []
        for mb in (2, 4):
            cfg = write_config(self.tmp, f"eval{mb}", micro_batch=mb, eval_windows=8, seed=5)
            proc = run("lm.train", "--config", str(cfg), "--device", "cpu", "--eval-only")[0]
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            losses.append(json.loads((self.tmp / f"eval{mb}" / "eval_only.json").read_text(encoding="utf-8"))["val_loss"])
        self.assertAlmostEqual(losses[0], losses[1], places=5)


class ContinueCorpusTests(unittest.TestCase):
    def test_continuation_starts_after_the_last_document_and_keeps_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "txt").mkdir()
            # More documents than one of prepare_text's batches (1,024), each run stopping after a whole batch.
            docs = [f"document number {i} " + "word " * (20 + i % 7) for i in range(3000)]
            (tmp / "txt" / "a.txt").write_text("\n\n".join(docs), encoding="utf-8")
            common = ["--source", "local", "--local-dir", str(tmp / "txt"), "--vocab", "300", "--tokenizer-docs", "100",
                      "--val-tokens", "500"]
            first = run("lm.prepare_text", *common, "--out", str(tmp / "first"), "--tokens", "3000")[0]
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            cont = run("lm.prepare_text", *common, "--out", str(tmp / "cont"), "--tokens", "3000",
                       "--continue-from", str(tmp / "first"))[0]
            self.assertEqual(cont.returncode, 0, cont.stderr[-2000:])
            a, b = json.loads((tmp / "first" / "state.json").read_text()), json.loads((tmp / "cont" / "meta.json").read_text())
            self.assertEqual((b["continued_from"]["next_row"], b["continued_from"]["train_tokens"]),
                             (a["next_row"], a["train_tokens"]))
            for name in ("val.bin", "tokenizer.json"):
                self.assertEqual((tmp / "first" / name).read_bytes(), (tmp / "cont" / name).read_bytes())
            self.assertGreater(b["train_tokens"], 0)
            # One longer corpus, same tokenizer, holds exactly the two train files end to end: no document twice.
            (tmp / "whole").mkdir()
            (tmp / "whole" / "tokenizer.json").write_bytes((tmp / "first" / "tokenizer.json").read_bytes())
            whole = run("lm.prepare_text", *common, "--out", str(tmp / "whole"),
                        "--tokens", str(a["train_tokens"] + 1))[0]
            self.assertEqual(whole.returncode, 0, whole.stderr[-2000:])
            joined = (tmp / "first" / "train.bin").read_bytes() + (tmp / "cont" / "train.bin").read_bytes()
            self.assertEqual((tmp / "whole" / "train.bin").read_bytes(), joined)
