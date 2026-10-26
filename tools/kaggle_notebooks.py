"""Write round 2's Kaggle notebooks, kaggle/<name>/<name>.ipynb and kernel-metadata.json, from the cells below.

Serves docs/KAGGLE_LM_PLAN.md (C9): every round-2 step runs on Kaggle, from the public repository at a pinned
commit, so a notebook is regenerated whenever the code it runs changes. Push the code first, then write the
notebooks pinned to that commit and commit them. KAGGLE_USERNAME in kernel-metadata.json is replaced by the
account's name in a copy that is pushed (`kaggle kernels push -p <copy>`, run from inside the copy on Windows).

    qwen_data           CPU   DS7 v2 and the DS6 subset for Qwen v2                     (the user's account)
    qwen3b_explain_v2   T4x2  Qwen v2 QLoRA on both GPUs, then every evaluation          (the user's account)
    text150m_data       CPU   the 152M model's continuation tokens (lm.prepare_text)     (the teammate's account)
    text150m_train      T4x2  fp16 check, benchmark, gate, continued pretraining chain  (the teammate's account)
    text150m_stage_data CPU   DS6 and DS7 v2 tokenized for the 152M stages               (with the 152M chain)
    text150m_stages     T4x2  chat and explanation stages, then every 152M evaluation    (with the 152M chain)
    assistant_random_eval T4x2  Qwen rounds 1-2 and 152M round 1 on the randomized test set (the user's account)

The 152M notebooks run on whichever account holds cos30049a2-text150m-base and the chain's outputs: a private
notebook's output cannot be read from another account (26 Oct: the teammate's account needs phone verification
first, so they run on the user's).

Usage:  python tools/kaggle_notebooks.py <commit>      (the full hash of a commit already on GitHub)
"""
import json
import sys
from pathlib import Path

REPO_DIR, COMMIT = Path(__file__).resolve().parents[1], sys.argv[1]


def nb(cells):
    return {"cells": [{"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if kind == "code" else {})} for kind, src in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = r'''
import contextlib, hashlib, json, os, shutil, subprocess, sys, threading, time
from importlib.metadata import version
from pathlib import Path

T0 = time.time()                           # the session's clock: Kaggle stops a session 12 h after it starts
SRC = Path("/tmp/COS30049A2")             # the code and the data it works on; not part of the output
WORK = Path("/kaggle/working")             # the notebook's output


def sh(*cmd, log=None, cwd=None, env=None):
    """Run a command, print its output as it comes, keep a copy in WORK/<log>; raise if it fails."""
    with contextlib.ExitStack() as stack:
        copy = stack.enter_context(open(WORK / log, "a", encoding="utf-8")) if log else None
        p = stack.enter_context(subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                 text=True, bufsize=1, env=env))
        for line in p.stdout:
            print(line, end="", flush=True)
            if copy:
                copy.write(line)
                copy.flush()
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Kaggle grants internet (and GPUs) only to phone-verified accounts, and otherwise switches them off silently.
import socket
try:
    socket.create_connection(("github.com", 443), timeout=10).close()
except OSError:
    raise SystemExit("No internet in this session: verify a phone number (kaggle.com -> Settings -> Phone "
                     "verification), switch Internet on in the notebook's Settings panel, and run again.")
'''

CLONE = r'''
if not SRC.exists():
    sh("git", "clone", "--quiet", REPO, str(SRC))
sh("git", "checkout", "--quiet", COMMIT, cwd=SRC)
sh(sys.executable, "-m", "pip", "install", "--quiet", *PINS)
os.chdir(SRC)
'''

DATA_NB = [
    ("markdown", r'''
# COS30049 A2 - Round 2 data for the Qwen 3B writer (Kaggle CPU session)

Builds the data of round 2's Qwen fine-tune (`docs/KAGGLE_LM_PLAN.md`, section 4) from the project repository
([Defaul119/COS30049A2](https://github.com/Defaul119/COS30049A2)) at a fixed commit, on a CPU session, which uses
no GPU quota:

* **DS7 v2**: about 40,000 explanation pairs from DS1 and every DS2 address, plus the glossary
  (`python -m explorer.cli build-ds7 --out data/processed/ds7_v2 --settings ds7v2`). The build stops if a prompt
  of round 1's frozen test set would be trained on. Its held-out glossary questions go to `glossary_test.jsonl`.
* **DS6 subset**: 10,000 smol-smoltalk conversations, holding out the same 200 retention conversations as round 1
  (checked byte for byte).

Every output file's SHA-256 goes into `data_manifest.json`. The GPU notebook `cos30049a2-qwen3b-explain-v2` reads
this notebook's output, checks each file against it, and the manifest is committed to the repository.

**Settings (right-hand panel):** Accelerator **None**, **Internet on**. Then **Save Version -> Save & Run All
(Commit)**. It takes about half an hour.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this build uses; recorded in data_manifest.json
# environment.yml's versions, so the saved scikit-learn models load as they were fitted
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2",
        "huggingface_hub==0.30.2", "tokenizers==0.21.1"]
RETENTION_SHA256 = "0f862f4692d1af224c836824a34748176f588dd5aec3eb3c569a6f8bd6794931"  # round 1's chat_retention.jsonl
'''),
    ("code", SETUP + CLONE),
    ("code", r'''
sh(sys.executable, "-m", "explorer.cli", "build-ds7", "--out", "data/processed/ds7_v2", "--settings", "ds7v2",
   log="build-ds7-v2.log")
sh(sys.executable, "-m", "lm.chat_subset", "--max-conversations", "10000", "--retention-start", "2500",
   "--out", "data/qwen_v2", log="chat-subset-v2.log")
got = sha256(SRC / "data" / "qwen_v2" / "chat_retention.jsonl")
assert got == RETENTION_SHA256, "the retention conversations differ from round 1's; stop and check"
print("retention conversations: same as round 1")
'''),
    ("code", r'''
# The data, the build's summary table and log, and a manifest the GPU notebook checks and the repository keeps.
files = sorted([*(SRC / "data" / "processed" / "ds7_v2").glob("*.json*"), *(SRC / "data" / "qwen_v2").glob("*.json*")])
extras = [SRC / "reports" / "tables" / "ds7v2_summary.csv", SRC / "reports" / "logs" / "build-ds7.log"]
for f in files + [e for e in extras if e.exists()]:
    dest = WORK / f.relative_to(SRC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dest)
manifest = {"commit": COMMIT, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS},
            "files": {f.relative_to(SRC).as_posix(): sha256(f) for f in files}}
(WORK / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 2**20:9.1f} MB  {p.relative_to(WORK)}")
'''),
]

QWEN_NB = [
    ("markdown", r'''
# COS30049 A2 - Explorer Assistant 3B, round 2: QLoRA fine-tune of Qwen2.5-3B-Instruct on both T4s

Round 2 of the Qwen writer (`docs/KAGGLE_LM_PLAN.md`): a fresh adapter on about four times round 1's explanation
data, the glossary repeated 4 times and 10,000 DS6 conversations (`configs/qwen3b_explain_v2.json`), at most 2 epochs,
stopping early when the validation loss stops improving. It trains on **both** GPUs of a T4 x2 session (torchrun,
`lm/dist.py`); round 1 used one, which left half of every quota hour idle.

**Settings (right-hand panel):** Accelerator **GPU T4 x2**, **Internet on**. **Inputs:** this account's notebooks
`cos30049a2-qwen-data` (the data) and `cos30049a2-qwen3b-explain` (round 1, whose adapter is also scored on the new
glossary test). Then **Save Version -> Save & Run All (Commit)**; the page can be closed.

**What it does:** clones the repository at `COMMIT`, installs `environment-gpu.yml`'s versions, copies the data from
the input and checks every file's SHA-256 against its `data_manifest.json`, benchmarks 5 steps, trains until done or
until the session's time is nearly up, and then, if at least `EVAL_HOURS` are left, evaluates with both GPUs at once:
round 2 on the frozen DS7 v1 test, DS10 and the glossary test, round 1 on the glossary test.

**If the session ends first:** add this notebook's own previous output as an input, set `RESUME_FROM` to its
`runs/qwen3b-explain-v2` folder, and run again: training continues from the last checkpoint (at most 20 minutes are
lost), or, if it had finished, only the evaluation runs.

**Bringing the result home:** `kaggle kernels output <owner>/cos30049a2-qwen3b-explain-v2 -p <dir>`, then copy
`runs/qwen3b-explain-v2/` to `runs/` and `reports/tables/assistant_eval*` to `reports/tables/`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses; recorded in kaggle_run.json
PINS = ["transformers==4.52.4", "peft==0.15.2", "bitsandbytes==0.48.1", "accelerate==1.7.0"]  # environment-gpu.yml
SESSION_HOURS = 11.75  # Kaggle ends a GPU session at 12 h; everything here finishes before this
EVAL_HOURS = 1.0       # what the evaluations need with both GPUs and EVAL_BATCH (training stops early enough)
EVAL_BATCH = 16        # prompts Qwen answers together
RESUME_FROM = None     # e.g. "/kaggle/input/<this notebook>/runs/qwen3b-explain-v2" to continue a stopped run
'''),
    ("code", SETUP + r'''
if shutil.which("nvidia-smi") is None:
    raise SystemExit("No GPU in this session: Settings panel -> Accelerator -> GPU T4 x2 (needs a verified phone).")
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
RUN = WORK / "runs" / "qwen3b-explain-v2"
''' + CLONE + r'''
sh("nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv")
print(f"{GPUS} GPU(s)")
'''),
    ("code", r'''
def find_input(rel):
    """The file under /kaggle/input whose path ends with rel (inputs mount under their own folder names)."""
    hits = [p for p in Path("/kaggle/input").rglob(Path(rel).name) if p.as_posix().endswith(rel)]
    if not hits:
        raise SystemExit(f"{rel} is not in any input: add the notebooks named at the top as inputs")
    return hits[0]


# The data notebook's output, file by file, checked against its manifest.
manifest_path = find_input("data_manifest.json")
data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for rel, digest in data_manifest["files"].items():
    dest = SRC / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_path.parent / rel, dest)
    assert sha256(dest) == digest, f"{rel} differs from data_manifest.json; stop and check"
print(f"{len(data_manifest['files'])} data files match data_manifest.json (built at commit {data_manifest['commit'][:7]})")

cfg = json.loads((SRC / "configs" / "qwen3b_explain_v2.json").read_text(encoding="utf-8"))
cfg["out_dir"] = str(RUN)
CONFIG = Path("/tmp/qwen3b_v2_kaggle.json")
CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
V1_ADAPTER = find_input("runs/qwen3b-explain/adapter_best/adapter_config.json").parent
if RESUME_FROM and not (RUN / "state.pt").exists():
    shutil.copytree(RESUME_FROM, RUN, dirs_exist_ok=True)
    print("resuming from", RESUME_FROM)
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(GPUS)]
'''),
    ("code", r'''
# 5 optimizer steps on the longest batches, on every GPU: speed and peak memory, nothing saved (skipped when resuming).
if not (RUN / "state.pt").exists():
    sh(*TORCHRUN, "-m", "lm.qlora", "--config", str(CONFIG), "--bench", log="qwen-v2-bench.log")
'''),
    ("code", r'''
def finished():
    meta_path = RUN / "meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta["steps_done"] >= meta["max_steps"] or bool(meta.get("stopped"))


# Train for the time the session has left, keeping EVAL_HOURS (and a margin for loading) for the evaluation.
hours = SESSION_HOURS - (time.time() - T0) / 3600 - EVAL_HOURS - 0.3
if not finished():
    sh(*TORCHRUN, "-m", "lm.qlora", "--config", str(CONFIG), "--max-hours", f"{hours:.2f}", log="qwen-v2-train.log")
print("training finished" if finished() else "training stopped for time: run again with RESUME_FROM (see the top)")
'''),
    ("code", r'''
# Every evaluation the adoption rule needs (docs/KAGGLE_LM_PLAN.md section 3), shared between the GPUs: one queue
# of commands per GPU, run side by side. Different test sets write different tables, so they never collide.
V1_CONFIG = SRC / "configs" / "qwen3b_explain.json"
GLOSSARY = SRC / "data" / "processed" / "ds7_v2" / "glossary_test.jsonl"
EVALS = [("qwen3b_v2", CONFIG, RUN / "adapter_best", SRC / "data" / "processed" / "ds7" / "test.jsonl", ""),
         ("qwen3b_v2", CONFIG, RUN / "adapter_best", SRC / "data" / "processed" / "ds10" / "explain.jsonl", "ds10"),
         ("qwen3b_v2", CONFIG, RUN / "adapter_best", GLOSSARY, "glossary"),
         ("qwen3b", V1_CONFIG, V1_ADAPTER, GLOSSARY, "glossary")]
# The last two write one table (assistant_eval_glossary.csv), so they share a queue.
QUEUES = [[EVALS[0]], EVALS[1:]] if GPUS > 1 else [EVALS]


def run_queue(gpu, queue):
    for model, config, adapter, test, tag in queue:
        sh(sys.executable, "-m", "lm.eval_explain", "--model", model, "--config", str(config), "--adapter", str(adapter),
           "--test", str(test), "--tag", tag, "--batch-size", str(EVAL_BATCH),
           log=f"eval-{model}-{tag or 'ds7'}.log", env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))


left = SESSION_HOURS - (time.time() - T0) / 3600
if not finished():
    print("not evaluated: training has not finished")
elif left < EVAL_HOURS:
    print(f"not evaluated: {left:.1f} h left; run again with RESUME_FROM and it only evaluates")
else:
    threads = [threading.Thread(target=run_queue, args=(gpu, q)) for gpu, q in enumerate(QUEUES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tables = WORK / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for f in (SRC / "reports" / "tables").glob("assistant_eval*.csv"):
        shutil.copy(f, tables)
'''),
    ("code", r'''
# What ran, where, with which versions: for the report's reproducibility notes.
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
meta = json.loads((RUN / "meta.json").read_text(encoding="utf-8")) if (RUN / "meta.json").exists() else {}
(WORK / "kaggle_run.json").write_text(json.dumps({
    "commit": COMMIT, "data_commit": data_manifest["commit"], "gpus": gpu, "python": sys.version.split()[0],
    "packages": {p: version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
    "finished": finished(), "resumed_from": RESUME_FROM, "session_hours": round((time.time() - T0) / 3600, 2),
    "meta": {k: meta.get(k) for k in ("steps_done", "max_steps", "best_val_loss", "stopped", "tok_per_s",
                                      "peak_memory_gb", "hardware", "fp16_loss_scaling")},
}, indent=2), encoding="utf-8")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 2**20:9.1f} MB  {p.relative_to(WORK)}")
'''),
]
FIND = r'''
def find_input(rel):
    """The file under /kaggle/input whose path ends with rel (inputs mount under their own folder names)."""
    hits = [p for p in Path("/kaggle/input").rglob(Path(rel).name) if p.as_posix().endswith(rel)]
    if not hits:
        raise SystemExit(f"{rel} is not in any input: add the inputs named at the top")
    return hits[0]


def check_base():
    """The round-1 files (dataset cos30049a2-text150m-base), each checked against the dataset's SHA256SUMS.txt."""
    sums = find_input("SHA256SUMS.txt")
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha256(sums.parent / name.lstrip("*")) == digest, f"{name} differs from SHA256SUMS.txt"
    return sums.parent
'''

TEXT_DATA_NB = [
    ("markdown", r'''
# COS30049 A2 - Round 2 pretraining tokens for the 152M model (Kaggle CPU session)

Continues round 1's FineWeb-Edu corpus (`docs/KAGGLE_LM_PLAN.md`, section 4, C4) on a CPU session, which uses no
GPU quota: `python -m lm.prepare_text --continue-from` starts at the document after round 1's last one (file 2, row
483,432 of the 10B-token sample), with round 1's tokenizer and validation tokens, so no document is trained on twice
and both rounds' validation losses are measured on the same tokens.

**Settings (right-hand panel):** Accelerator **None**, **Internet on**. **Input:** this account's dataset
`cos30049a2-text150m-base` (round 1's tokenizer, `val.bin`, `meta.json` and `state.json`, with SHA-256). Then **Save
Version -> Save & Run All (Commit)**. Output: `text_cont/` (about 2 bytes per token) and `data_manifest.json`, read by
`cos30049a2-text150m-train`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this build uses; recorded in data_manifest.json
TOKENS = 3_000_000_000  # new training tokens: more than the GPU quota can train this round, so the gate decides
PINS = ["numpy==2.2.6", "tokenizers==0.21.1", "huggingface_hub==0.30.2", "pyarrow==20.0.0"]  # environment.yml
'''),
    ("code", SETUP + CLONE + FIND),
    ("code", r'''
base = check_base()
(SRC / "data" / "text").mkdir(parents=True, exist_ok=True)
for name in ("tokenizer.json", "val.bin", "meta.json", "state.json"):
    shutil.copy(base / name, SRC / "data" / "text" / name)
OUT = WORK / "text_cont"
sh(sys.executable, "-m", "lm.prepare_text", "--continue-from", "data/text", "--out", str(OUT), "--tokens", str(TOKENS),
   "--raw", "/tmp/fineweb-edu", "--delete-raw", log="prepare-text-cont.log")
'''),
    ("code", r'''
meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
manifest = {"commit": COMMIT, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS},
            "train_tokens": meta["train_tokens"], "continued_from": meta["continued_from"],
            "files": {f"text_cont/{f.name}": sha256(f) for f in sorted(OUT.iterdir()) if f.is_file()}}
(WORK / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
'''),
]

TEXT_TRAIN_NB = [
    ("markdown", r'''
# COS30049 A2 - Round 2: continued pretraining of the 152M model on both T4s

Continues round 1's 152M model (validation loss 3.1305 after 2.0B tokens) on new FineWeb-Edu tokens
(`configs/text_150m_pretrain_cont.json`, `docs/KAGGLE_LM_PLAN.md`), with a warm-up, stable, decay schedule, on both
GPUs of a T4 x2 session (torchrun, `lm/dist.py`). Pretraining is one chain of sessions: each resumes from the
previous one's checkpoint.

**Settings:** Accelerator **GPU T4 x2**, **Internet on**. **Inputs:** this account's dataset
`cos30049a2-text150m-base` and notebook `cos30049a2-text150m-data`; from the second session on, also this notebook's
own previous output, with `RESUME_FROM` set below. Then **Save Version -> Save & Run All (Commit)**.

**Session 1** checks that the round-1 weights give the same validation loss in fp16 as in bf16 (T4s have no bf16;
within `FP16_TOLERANCE` of 3.1305), benchmarks micro-batch sizes on both GPUs, and sets `total_tokens` by the gate:
the tokens the measured speed can train in `GPU_HOURS` (docs/KAGGLE_LM_PLAN.md section 6), saved in `gate.json` so
later sessions keep it. **Every session** then trains until the session's time is nearly up, saving a checkpoint
every 30 minutes. To train for longer (another account's quota), raise `TOTAL_TOKENS` before the decay starts (the
last 15%).
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses; recorded in kaggle_run.json
SESSION_HOURS = 11.75  # Kaggle ends a GPU session at 12 h
GPU_HOURS = 50.0       # the gate: GPU hours this chain may use in all: this week's quota and the next (reset Sat
                       # 07:00 GMT+7) on this account, less about 6 h kept for the chat and explanation stages
TOTAL_TOKENS = None    # None: session 1's gate decides; a number overrides gate.json (only before the decay starts)
ROUND1_VAL_LOSS = 3.1305  # runs/text-150m/log.jsonl, last evaluation (bf16 on the RTX 3060, the same 160 windows)
FP16_TOLERANCE = 0.01  # docs/KAGGLE_LM_PLAN.md section 5, the fp16 check
RESUME_FROM = None     # e.g. "/kaggle/input/<this notebook>/runs/text-150m-cont" from the second session on
PINS = ["tokenizers==0.21.1", "numpy==2.2.6"]  # environment.yml; torch is Kaggle's own (recorded)
'''),
    ("code", SETUP + r'''
if shutil.which("nvidia-smi") is None:
    raise SystemExit("No GPU in this session: Settings panel -> Accelerator -> GPU T4 x2 (needs a verified phone).")
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
RUN = WORK / "runs" / "text-150m-cont"
''' + CLONE + FIND + r'''
sh("nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv")
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(GPUS)]
'''),
    ("code", r'''
base = check_base()
data_manifest = json.loads(find_input("data_manifest.json").read_text(encoding="utf-8"))
DATA = find_input("text_cont/meta.json").parent
for rel, digest in data_manifest["files"].items():
    if not rel.endswith("train.bin"):  # 6 GB: its size is checked instead of its hash, which would take minutes
        assert sha256(DATA.parent / rel) == digest, f"{rel} differs from data_manifest.json"
assert (DATA / "train.bin").stat().st_size == 2 * data_manifest["train_tokens"], "train.bin is incomplete"
if RESUME_FROM and not (RUN / "ckpt_last.pt").exists():
    shutil.copytree(RESUME_FROM, RUN, dirs_exist_ok=True)
    print("resuming from", RESUME_FROM)
RUN.mkdir(parents=True, exist_ok=True)


def write_config(**override):
    cfg = json.loads((SRC / "configs" / "text_150m_pretrain_cont.json").read_text(encoding="utf-8"))
    cfg.update({"data_dir": str(DATA), "out_dir": str(RUN), "init_from": str(base / "text150m_base.pt"), **override})
    path = Path("/tmp/text150m_cont.json")
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path
'''),
    ("code", r'''
gate_path = RUN / "gate.json"
if not gate_path.exists():
    # The fp16 check: round 1 trained in bf16, which T4s lack; fp16 overflows sooner, so the starting point must match.
    sh(*TORCHRUN, "-m", "lm.train", "--config", str(write_config()), "--eval-only", log="fp16-check.log")
    fp16 = json.loads((RUN / "eval_only.json").read_text(encoding="utf-8"))["val_loss"]
    print(f"fp16 validation loss {fp16:.4f} against round 1's {ROUND1_VAL_LOSS} (bf16)")
    if abs(fp16 - ROUND1_VAL_LOSS) > FP16_TOLERANCE:
        raise SystemExit("fp16 check failed: do not train in fp16 (docs/KAGGLE_LM_PLAN.md section 7)")
    sh(*TORCHRUN, "-m", "lm.train", "--config", str(write_config()), "--bench", log="bench.log")
    bench = json.loads((RUN / "bench.json").read_text(encoding="utf-8"))
    rate = bench["tok_per_s_per_gpu"] * GPUS
    # The gate (plan section 6): what the measured speed trains in GPU_HOURS, less 10% for evaluations and handoffs,
    # in whole optimizer steps, and no more than the tokens prepared.
    batch = json.loads((SRC / "configs" / "text_150m_pretrain_cont.json").read_text(encoding="utf-8"))["batch_tokens"]
    total = min(int(rate * GPU_HOURS * 3600 * 0.9), data_manifest["train_tokens"]) // batch * batch
    gate_path.write_text(json.dumps({"micro_batch": bench["micro_batch"], "tok_per_s": rate, "gpus": GPUS,
                                     "gpu_hours": GPU_HOURS, "total_tokens": total, "fp16_val_loss": fp16},
                                    indent=2) + "\n", encoding="utf-8")
gate = json.loads(gate_path.read_text(encoding="utf-8"))
if TOTAL_TOKENS:
    gate["total_tokens"] = TOTAL_TOKENS
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
print(json.dumps(gate, indent=2))
'''),
    ("code", r'''
config = write_config(micro_batch=gate["micro_batch"], total_tokens=gate["total_tokens"])
hours = SESSION_HOURS - (time.time() - T0) / 3600 - 0.3  # a margin for the final checkpoint and evaluation
sh(*TORCHRUN, "-m", "lm.train", "--config", str(config), "--max-hours", f"{hours:.2f}", log="train.log")
'''),
    ("code", r'''
# What ran, where, with which versions; and the run folder's size (the output keeps at most 20 GB).
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
last = [json.loads(l) for l in (RUN / "log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
evals = [r for r in last if "val_loss" in r]
(WORK / "kaggle_run.json").write_text(json.dumps({
    "commit": COMMIT, "data_commit": data_manifest["commit"], "gpus": gpu, "python": sys.version.split()[0],
    "packages": {p: version(p) for p in ("torch", "numpy", "tokenizers")}, "gate": gate, "resumed_from": RESUME_FROM,
    "session_hours": round((time.time() - T0) / 3600, 2), "last_step": max(r.get("step", 0) for r in last),
    "last_eval": evals[-1] if evals else None, "finished": (RUN / "model.pt").exists(),
}, indent=2), encoding="utf-8")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 2**20:9.1f} MB  {p.relative_to(WORK)}")
'''),
]


STAGE_DATA_NB = [
    ("markdown", r'''
# COS30049 A2 - Round 2 data for the 152M model's chat and explanation stages (Kaggle CPU session)

Tokenizes, with the 152M model's own tokenizer, what its two fine-tuning stages train on (`docs/KAGGLE_LM_PLAN.md`,
section 4): DS6 smol-smoltalk for the chat stage, as in round 1, and DS7 v2 for the explanation stage, with the
glossary's training rows repeated 4 times as for Qwen v2. CPU only, so no GPU quota.

**Settings:** Accelerator **None**, **Internet on**. **Inputs:** this account's dataset `cos30049a2-text150m-base`
(the tokenizer) and notebook `cos30049a2-qwen-data` (DS7 v2). Output `stage/` and `stage_manifest.json`, read by
`cos30049a2-text150m-stages`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this build uses; recorded in stage_manifest.json
PINS = ["numpy==2.2.6", "tokenizers==0.21.1", "huggingface_hub==0.30.2", "pyarrow==20.0.0"]  # environment.yml
GLOSSARY_REPEATS = 4   # as configs/qwen3b_explain_v2.json -> repeat_sources
'''),
    ("code", SETUP + CLONE + FIND),
    ("code", r'''
base = check_base()
TOK = base / "tokenizer.json"
qwen_manifest_path = find_input("data_manifest.json")
qwen_manifest = json.loads(qwen_manifest_path.read_text(encoding="utf-8"))
DS7 = SRC / "data" / "processed" / "ds7_v2"
DS7.mkdir(parents=True, exist_ok=True)
for name in ("train.jsonl", "val.jsonl", "glossary_test.jsonl"):
    rel = f"data/processed/ds7_v2/{name}"
    shutil.copy(qwen_manifest_path.parent / rel, DS7 / name)
    assert sha256(DS7 / name) == qwen_manifest["files"][rel], f"{rel} differs from data_manifest.json"
# The glossary rows are in train.jsonl once; GLOSSARY_REPEATS - 1 more copies go in as extra files.
glossary = Path("/tmp/glossary_train.jsonl")
with open(DS7 / "train.jsonl", encoding="utf-8") as f, open(glossary, "w", encoding="utf-8") as out:
    out.writelines(line for line in f if json.loads(line)["source"] == "glossary")
STAGE = WORK / "stage"
sh(sys.executable, "-m", "lm.prepare_chat", "--tokenizer", str(TOK), "--out", str(STAGE / "chat"), log="prepare-chat.log")
sh(sys.executable, "-m", "lm.prepare_chat", "--tokenizer", str(TOK), "--source", "jsonl", "--jsonl", str(DS7 / "train.jsonl"),
   "--val-jsonl", str(DS7 / "val.jsonl"), "--extra", *[str(glossary)] * (GLOSSARY_REPEATS - 1),
   "--out", str(STAGE / "ds7_v2_chat"), log="prepare-ds7-v2-chat.log")
shutil.copy(DS7 / "glossary_test.jsonl", STAGE / "glossary_test.jsonl")
'''),
    ("code", r'''
manifest = {"commit": COMMIT, "ds7_v2_commit": qwen_manifest["commit"],
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {p.relative_to(WORK).as_posix(): sha256(p) for p in sorted(STAGE.rglob("*")) if p.is_file()}}
(WORK / "stage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 2**20:9.1f} MB  {p.relative_to(WORK)}")
'''),
]

STAGES_NB = [
    ("markdown", r'''
# COS30049 A2 - Round 2: the 152M model's chat and explanation stages, then every evaluation (T4 x2)

After the continued pretraining (`cos30049a2-text150m-train`), round 1's two fine-tuning stages run again from the
new weights on both GPUs: chat on DS6 (`configs/text_150m_chat_v2.json`), then explanations on DS7 v2
(`configs/text_150m_explain_v2.json`). The new writer is then scored on the frozen DS7 v1 test, DS10 and the glossary
test, and round 1's writer on the glossary test, side by side on the two GPUs (`docs/KAGGLE_LM_PLAN.md` section 3).

**Settings:** Accelerator **GPU T4 x2**, **Internet on**. **Inputs:** this account's dataset `cos30049a2-text150m-base`
and notebooks `cos30049a2-text150m-stage-data` and `cos30049a2-text150m-train` (its finished `model.pt`); to resume,
also this notebook's own previous output, with `RESUME_FROM` set.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses; recorded in kaggle_run.json
SESSION_HOURS = 11.75  # Kaggle ends a GPU session at 12 h
EVAL_HOURS = 0.75      # what the evaluations need on two GPUs
RESUME_FROM = None     # e.g. "/kaggle/input/<this notebook>/runs" to continue a stopped session
PINS = ["tokenizers==0.21.1", "numpy==2.2.6"]  # environment.yml; torch is Kaggle's own (recorded)
'''),
    ("code", SETUP + r'''
if shutil.which("nvidia-smi") is None:
    raise SystemExit("No GPU in this session: Settings panel -> Accelerator -> GPU T4 x2 (needs a verified phone).")
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
RUNS = WORK / "runs"
''' + CLONE + FIND + r'''
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(GPUS)]
'''),
    ("code", r'''
base = check_base()
stage_manifest_path = find_input("stage_manifest.json")
for rel, digest in json.loads(stage_manifest_path.read_text(encoding="utf-8"))["files"].items():
    assert sha256(stage_manifest_path.parent / rel) == digest, f"{rel} differs from stage_manifest.json"
STAGE = stage_manifest_path.parent / "stage"
PRETRAINED = find_input("runs/text-150m-cont/model.pt")
if RESUME_FROM and not RUNS.exists():
    shutil.copytree(RESUME_FROM, RUNS)
    print("resuming from", RESUME_FROM)


def config(name, **paths):
    cfg = json.loads((SRC / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    cfg.update({k: str(v) for k, v in paths.items()})
    path = Path(f"/tmp/{name}.json")
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


CHAT, EXPLAIN = RUNS / "text-150m-chat-v2", RUNS / "text-150m-explain-v2"
STAGES = [(config("text_150m_chat_v2", data_dir=STAGE / "chat", out_dir=CHAT, init_from=PRETRAINED), CHAT),
          (config("text_150m_explain_v2", data_dir=STAGE / "ds7_v2_chat", out_dir=EXPLAIN,
                  init_from=CHAT / "model.pt"), EXPLAIN)]
'''),
    ("code", r'''
for cfg, out in STAGES:
    if (out / "model.pt").exists():
        continue
    hours = SESSION_HOURS - (time.time() - T0) / 3600 - EVAL_HOURS - 0.2
    sh(*TORCHRUN, "-m", "lm.train", "--config", str(cfg), "--max-hours", f"{hours:.2f}", log=f"{out.name}.log")
    if not (out / "model.pt").exists():
        print(f"{out.name} stopped for time: run again with RESUME_FROM (see the top)")
        break
'''),
    ("code", r'''
# Every evaluation the adoption rule needs, one queue per GPU; each test set writes its own tables.
V2 = EXPLAIN / "model_best.pt"
GLOSSARY = STAGE / "glossary_test.jsonl"
RANDOM = SRC / "data" / "processed" / "random_test"
EVALS = [("text150m_v2", V2, SRC / "data" / "processed" / "ds7" / "test.jsonl", ""),
         ("text150m_v2", V2, RANDOM / "real.jsonl", "random_real"),
         ("text150m_v2", V2, SRC / "data" / "processed" / "ds10" / "explain.jsonl", "ds10"),
         ("text150m_v2", V2, GLOSSARY, "glossary"),
         ("text150m", base / "text150m_explain_v1.pt", GLOSSARY, "glossary"),
         ("text150m_v2", V2, RANDOM / "synthetic.jsonl", "random_synthetic")]
# A test set's evaluations stay in one queue: they write one summary table.
QUEUES = [EVALS[:2], EVALS[2:]] if GPUS > 1 else [EVALS]


def run_queue(gpu, queue):
    for model, checkpoint, test, tag in queue:
        sh(sys.executable, "-m", "lm.eval_explain", "--model", model, "--checkpoint", str(checkpoint), "--test", str(test),
           "--tag", tag, log=f"eval-{model}-{tag or 'ds7'}.log", env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))


if (EXPLAIN / "model.pt").exists():
    threads = [threading.Thread(target=run_queue, args=(gpu, q)) for gpu, q in enumerate(QUEUES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tables = WORK / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for f in (SRC / "reports" / "tables").glob("assistant_eval*.csv"):
        shutil.copy(f, tables)
(WORK / "kaggle_run.json").write_text(json.dumps({
    "commit": COMMIT, "gpus": GPUS, "python": sys.version.split()[0],
    "packages": {p: version(p) for p in ("torch", "numpy", "tokenizers")}, "resumed_from": RESUME_FROM,
    "session_hours": round((time.time() - T0) / 3600, 2),
    "finished": {out.name: (out / "model.pt").exists() for _, out in STAGES},
}, indent=2), encoding="utf-8")
'''),
]


RANDOM_EVAL_NB = [
    ("markdown", r'''
# COS30049 A2 - The randomized test set: both Explorer Assistants, both rounds (T4 x2)

Scores Qwen 3B (round 2 and round 1) and the 152M model (round 1) on the randomized test set
(`data/processed/random_test/`, `explorer/random_test.py`): 500 real prompts drawn from DS7 v2's held-out addresses
and 500 synthetic transactions no writer has seen. The template's own rows were scored when the set was built; the
152M model's round 2 is scored by `cos30049a2-text150m-stages`. One GPU takes the real prompts, the other the
synthetic ones, so each summary table has one writer process.

**Settings:** Accelerator **GPU T4 x2**, **Internet on**. **Inputs:** this account's notebooks
`cos30049a2-qwen3b-explain-v2` (round 2's adapter) and `cos30049a2-qwen3b-explain` (round 1's), and dataset
`cos30049a2-text150m-base` (the 152M round-1 writer). Then **Save Version -> Save & Run All (Commit)**; about an hour.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code and the test set this run uses; recorded in kaggle_run.json
PINS = ["transformers==4.52.4", "peft==0.15.2", "bitsandbytes==0.48.1", "accelerate==1.7.0", "tokenizers==0.21.1"]
EVAL_BATCH = 16       # prompts Qwen answers together
'''),
    ("code", SETUP + r'''
if shutil.which("nvidia-smi") is None:
    raise SystemExit("No GPU in this session: Settings panel -> Accelerator -> GPU T4 x2 (needs a verified phone).")
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
''' + CLONE + FIND),
    ("code", r'''
base = check_base()
V2_ADAPTER = find_input("runs/qwen3b-explain-v2/adapter_best/adapter_config.json").parent
V1_ADAPTER = find_input("runs/qwen3b-explain/adapter_best/adapter_config.json").parent
RANDOM = SRC / "data" / "processed" / "random_test"
manifest = json.loads((RANDOM / "manifest.json").read_text(encoding="utf-8"))
for f in manifest["files"]:
    assert sha256(SRC / f["path"]) == f["sha256"], f"{f['path']} differs from its manifest"
WRITERS = [("qwen3b_v2", ["--config", str(SRC / "configs" / "qwen3b_explain_v2.json"), "--adapter", str(V2_ADAPTER),
                          "--batch-size", str(EVAL_BATCH)]),
           ("qwen3b", ["--config", str(SRC / "configs" / "qwen3b_explain.json"), "--adapter", str(V1_ADAPTER),
                       "--batch-size", str(EVAL_BATCH)]),
           ("text150m", ["--checkpoint", str(base / "text150m_explain_v1.pt")])]
QUEUES = [[(w, a, "real") for w, a in WRITERS], [(w, a, "synthetic") for w, a in WRITERS]]
if GPUS == 1:
    QUEUES = [QUEUES[0] + QUEUES[1]]


def run_queue(gpu, queue):
    for model, args, part in queue:
        sh(sys.executable, "-m", "lm.eval_explain", "--model", model, *args, "--test", str(RANDOM / f"{part}.jsonl"),
           "--tag", f"random_{part}", log=f"eval-{model}-random_{part}.log",
           env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))


threads = [threading.Thread(target=run_queue, args=(gpu, q)) for gpu, q in enumerate(QUEUES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
tables = WORK / "reports" / "tables"
tables.mkdir(parents=True, exist_ok=True)
for f in (SRC / "reports" / "tables").glob("assistant_eval*random*.csv"):
    shutil.copy(f, tables)
(WORK / "kaggle_run.json").write_text(json.dumps({
    "commit": COMMIT, "gpus": GPUS, "python": sys.version.split()[0], "session_hours": round((time.time() - T0) / 3600, 2),
    "packages": {p: version(p) for p in ("torch", "transformers", "peft", "bitsandbytes", "tokenizers")},
}, indent=2), encoding="utf-8")
for p in sorted(tables.iterdir()):
    print(p.name)
'''),
]


def metadata(slug, code_file, gpu, kernel_sources, dataset_sources=()):
    return {"id": f"KAGGLE_USERNAME/{slug}", "title": slug, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": list(dataset_sources), "competition_sources": [],
            "kernel_sources": kernel_sources, "model_sources": []}


BASE = ["KAGGLE_USERNAME/cos30049a2-text150m-base"]
for folder, slug, cells, gpu, sources, datasets in [
        ("qwen_data", "cos30049a2-qwen-data", DATA_NB, False, [], []),
        ("qwen3b_explain_v2", "cos30049a2-qwen3b-explain-v2", QWEN_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-qwen-data", "KAGGLE_USERNAME/cos30049a2-qwen3b-explain"], []),
        ("text150m_data", "cos30049a2-text150m-data", TEXT_DATA_NB, False, [], BASE),
        ("text150m_train", "cos30049a2-text150m-train", TEXT_TRAIN_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-text150m-data"], BASE),
        ("text150m_stage_data", "cos30049a2-text150m-stage-data", STAGE_DATA_NB, False,
         ["KAGGLE_USERNAME/cos30049a2-qwen-data"], BASE),
        ("text150m_stages", "cos30049a2-text150m-stages", STAGES_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-text150m-stage-data", "KAGGLE_USERNAME/cos30049a2-text150m-train"], BASE),
        ("assistant_random_eval", "cos30049a2-assistant-random-eval", RANDOM_EVAL_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-qwen3b-explain-v2", "KAGGLE_USERNAME/cos30049a2-qwen3b-explain"], BASE)]:
    out = REPO_DIR / "kaggle" / folder
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{folder}.ipynb").write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata(slug, f"{folder}.ipynb", gpu, sources, datasets), indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
