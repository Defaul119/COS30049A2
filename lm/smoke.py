"""End-to-end check of the whole text pipeline on a tiny model, in a minute or two.

Run this on the training machine before starting the multi-day run. It uses a
made-up corpus (no downloads) and goes through every stage with the real
commands: tokenizer, data files, pretraining, stopping and resuming, chat data,
chat fine-tuning and replying. It prints PASS or FAIL for each stage.

Usage:
    python -m lm.smoke            # files go to runs/smoke_test and are deleted afterwards
    python -m lm.smoke --keep     # keep them for inspection
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

NOUNS = ["river", "mountain", "teacher", "student", "computer", "garden", "city", "planet", "library", "engine",
         "forest", "doctor", "market", "bridge", "ocean", "village", "network", "museum", "farmer", "camera"]
ADJECTIVES = ["large", "quiet", "bright", "old", "modern", "small", "busy", "green", "cold", "famous"]
VERBS = ["visits", "describes", "builds", "protects", "studies", "crosses", "repairs", "watches", "measures", "follows"]
COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "grey", "brown"]


def make_corpus(folder: Path, rng: random.Random) -> None:
    """A few hundred paragraphs of simple, patterned English."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in range(4):
        paragraphs = []
        for _ in range(300):
            sentences = [f"The {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {rng.choice(VERBS)} "
                         f"the {rng.choice(NOUNS)}." for _ in range(rng.randint(3, 8))]
            paragraphs.append(" ".join(sentences))
        (folder / f"part{f}.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")


def make_chats(path: Path, rng: random.Random) -> None:
    """Conversations with a fixed answer per question, so fine-tuning has something learnable."""
    colour_of = {n: COLOURS[i % len(COLOURS)] for i, n in enumerate(NOUNS)}
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(600):
            n = rng.choice(NOUNS)
            msgs = [{"role": "user", "content": f"What colour is the {n}?"},
                    {"role": "assistant", "content": f"The {n} is {colour_of[n]}."}]
            fh.write(json.dumps({"messages": msgs}) + "\n")


def run(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", *args]
    print("  $ python -m " + " ".join(args), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    root = Path("runs/smoke_test")
    shutil.rmtree(root, ignore_errors=True)
    rng = random.Random(0)
    make_corpus(root / "corpus", rng)
    make_chats(root / "chats.jsonl", rng)

    tiny = {"seq_len": 128, "n_layer": 2, "n_head": 4, "d_model": 128, "mlp_hidden": 344}
    pre_cfg = {"name": "smoke", "data_dir": str(root / "text"), "out_dir": str(root / "run"), "model": tiny,
               "total_tokens": 160_000, "batch_tokens": 4096, "micro_batch": 8, "lr": 3e-3, "min_lr": 3e-4,
               "warmup_steps": 5, "eval_every": 10, "eval_batches": 4, "ckpt_every_minutes": 60,
               "bench_micro_batches": [16, 8, 4]}
    chat_cfg = {"name": "smoke-chat", "data_dir": str(root / "chat"), "out_dir": str(root / "chat_run"),
                "init_from": str(root / "run" / "model.pt"), "epochs": 30, "batch_tokens": 2048, "micro_batch": 8,
                "lr": 2e-3, "min_lr": 2e-4, "warmup_steps": 5, "weight_decay": 0.0, "eval_every": 10,
                "eval_batches": 4, "ckpt_every_minutes": 60}
    (root / "pre.json").write_text(json.dumps(pre_cfg), encoding="utf-8")
    (root / "chat.json").write_text(json.dumps(chat_cfg), encoding="utf-8")
    dev = ["--device", args.device]

    results = []

    def stage(name, proc, check=lambda: True):
        ok = proc.returncode == 0
        detail = ""
        if ok:
            try:
                check_result = check()
                ok, detail = (check_result if isinstance(check_result, tuple) else (bool(check_result), ""))
            except Exception as e:  # report a failed check instead of crashing the whole smoke test
                ok, detail = False, f"check failed: {e}"
        else:
            detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
            detail = detail[0]
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{' - ' + detail if detail else ''}\n", flush=True)
        results.append(ok)
        if not ok:
            print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
        return ok

    t0 = time.time()
    print("\n1. Tokenizer and token files")
    if not stage("prepare_text", run(["lm.prepare_text", "--source", "local", "--local-dir", str(root / "corpus"),
                                      "--out", str(root / "text"), "--vocab", "1000", "--tokenizer-docs", "200",
                                      "--val-tokens", "4000", "--tokens", "1000000000"]),
                 lambda: json.loads((root / "text" / "meta.json").read_text())["train_tokens"] > 20_000):
        sys.exit(1)

    print("2. Speed and memory test")
    stage("bench", run(["lm.train", "--config", str(root / "pre.json"), "--bench", *dev]))

    print("3. Pretraining, stopped part-way, then resumed")
    stage("train (first part)", run(["lm.train", "--config", str(root / "pre.json"), "--max-steps", "15", *dev]),
          lambda: (root / "run" / "ckpt_last.pt").exists())
    proc = run(["lm.train", "--config", str(root / "pre.json"), *dev])

    def pretrain_check():
        if "resuming from step 15" not in proc.stdout:
            return False, "did not resume from step 15"
        evals = [r["val_loss"] for r in read_log(root / "run" / "log.jsonl") if "val_loss" in r]
        if not (root / "run" / "model.pt").exists():
            return False, "model.pt missing"
        return evals[-1] < evals[0] - 1.0, f"validation loss {evals[0]:.2f} -> {evals[-1]:.2f}"
    stage("train (resumed to the end)", proc, pretrain_check)

    print("4. Text generation from the pretrained model")
    proc = run(["lm.generate", "--model", str(root / "run" / "model.pt"), "--prompt", "The quiet river",
                "--max-new-tokens", "30", "--seed", "0", *dev])
    stage("generate", proc, lambda: (len(proc.stdout.split()) > 5, proc.stdout.strip()[:120]))

    print("5. Chat data and chat fine-tuning")
    stage("prepare_chat", run(["lm.prepare_chat", "--source", "jsonl", "--jsonl", str(root / "chats.jsonl"),
                               "--tokenizer", str(root / "text" / "tokenizer.json"), "--out", str(root / "chat"),
                               "--seq-len", "128", "--val-every", "20"]))
    proc = run(["lm.train", "--config", str(root / "chat.json"), *dev])

    def chat_check():
        evals = [r["val_loss"] for r in read_log(root / "chat_run" / "log.jsonl") if "val_loss" in r]
        return evals[-1] < 0.5 * evals[0], f"validation loss {evals[0]:.2f} -> {evals[-1]:.2f}"
    stage("chat fine-tuning", proc, chat_check)

    print("6. A chat reply")
    sys.path.insert(0, str(Path.cwd()))
    from lm.chat import reply
    from lm.generate import load_model
    model, tok = load_model(root / "chat_run" / "model.pt", "cpu" if args.device == "auto" else args.device)
    answer = reply(model, tok, [{"role": "user", "content": "What colour is the river?"}], max_new_tokens=20,
                   temperature=0)
    expected = "The river is red."
    ok = answer == expected
    print(f"  {'PASS' if ok else 'FAIL'}  reply - asked 'What colour is the river?', got {answer!r} "
          f"(expected {expected!r})\n")
    results.append(ok)

    print(f"{sum(results)}/{len(results)} stages passed in {time.time() - t0:.0f}s")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
