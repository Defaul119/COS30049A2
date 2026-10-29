"""Write round 2's Kaggle notebooks, kaggle/<name>/<name>.ipynb and kernel-metadata.json, from the cells below.

Serves docs/KAGGLE_LM_PLAN.md (C9): every round-2 step runs on Kaggle, from the public repository at a pinned
commit, so a notebook is regenerated whenever the code it runs changes. Push the code first, then write the
notebooks pinned to that commit and commit them. KAGGLE_USERNAME in kernel-metadata.json is replaced by the
account's name in a copy that is pushed (`kaggle kernels push -p <copy>`, run from inside the copy on Windows).

    ds7_data            CPU   DS7 v2 (explanation pairs and the glossary)                (with the 152M chain)
    text150m_data       CPU   the 152M model's continuation tokens (lm.prepare_text)     (the teammate's account)
    text150m_train      T4x2  fp16 check, benchmark, gate, continued pretraining chain  (the teammate's account)
    text150m_stage_data CPU   DS6 and DS7 v2 tokenized for the 152M stages               (with the 152M chain)
    text150m_stages     T4x2  chat and explanation stages, then every 152M evaluation    (with the 152M chain)
    text150m_b3_trial   T4x2  stage B3 on round 1: explanation stage with and without weak cases, dev tier
    distribution        CPU   every scoring method over many plant seeds                 (any account)
    class_distribution  CPU   classifier metrics over seeds and a test bootstrap         (any account)
    stage_cpu           CPU   one TRAINING_PLAN stage: refit before / after, dev gate    (any account)
    stage_search        CPU   TRAINING_PLAN stages C1-C6: each method's grid, then the gate (any account)
    ensemble            CPU   TRAINING_PLAN stage D6: Elliptic classifier ensemble, gated (any account)
    stage_test          CPU   the test tier of the accepted DS1 stages, the selection rule (any account)

Only models the team trains from scratch are trained here (the tutor, 27 Oct 2026); the Qwen 3B notebooks of
26-27 Oct are on the qwen-archive branch.

The 152M notebooks run on whichever account holds cos30049a2-text150m-base and the chain's outputs: a private
notebook's output cannot be read from another account (26 Oct: the teammate's account needs phone verification
first, so they run on the user's).

Usage:  python tools/kaggle_notebooks.py <commit> [folder ...]   (the full hash of a commit already on GitHub;
        with folders, only those notebooks are rewritten, so the others keep their pins)
"""
import json
import sys
from pathlib import Path

REPO_DIR, COMMIT, ONLY = Path(__file__).resolve().parents[1], sys.argv[1], set(sys.argv[2:])


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

DS7_DATA_NB = [
    ("markdown", r'''
# COS30049 A2 - Round 2 explanation data (DS7 v2) for the 152M model (Kaggle CPU session)

Builds DS7 v2 (`docs/KAGGLE_LM_PLAN.md`, section 4) from the project repository
([Defaul119/COS30049A2](https://github.com/Defaul119/COS30049A2)) at a fixed commit, on a CPU session, which uses
no GPU quota: about 40,000 explanation pairs from DS1 and every DS2 address, plus the glossary
(`python -m explorer.cli build-ds7 --out data/processed/ds7_v2 --settings ds7v2`). The build stops if a prompt
of round 1's frozen test set would be trained on. Its held-out glossary questions go to `glossary_test.jsonl`.

Every output file's SHA-256 goes into `data_manifest.json`; `cos30049a2-text150m-stage-data` reads this notebook's
output and checks each file against it.

**Settings (right-hand panel):** Accelerator **None**, **Internet on**. Then **Save Version -> Save & Run All
(Commit)**. It takes about half an hour.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this build uses; recorded in data_manifest.json
# environment.yml's versions, so the saved scikit-learn models load as they were fitted
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2",
        "huggingface_hub==0.30.2", "tokenizers==0.21.1"]
'''),
    ("code", SETUP + CLONE),
    ("code", r'''
sh(sys.executable, "-m", "explorer.cli", "build-ds7", "--out", "data/processed/ds7_v2", "--settings", "ds7v2",
   log="build-ds7-v2.log")
'''),
    ("code", r'''
# The data, the build's summary table and log, and a manifest the stage-data notebook checks.
files = sorted((SRC / "data" / "processed" / "ds7_v2").glob("*.json*"))
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
RESUME_FROM = None     # from the second session on: "auto" (the one input holding text-150m-cont/ckpt_last.pt,
                       # wherever Kaggle mounts it) or that folder's path
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
if RESUME_FROM == "auto":  # Kaggle's mount layout changed once (/kaggle/input/datasets/<user>/...); search instead
    RESUME_FROM = str(find_input("text-150m-cont/ckpt_last.pt").parent)
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
glossary's training rows repeated 4 times; and, for stage B3 (`docs/TRAINING_PLAN.md`), the same plus the weak-case
examples (`ds7_v2_weak_chat`) and the writer's dev files (`dev/`). CPU only, so no GPU quota.

**Settings:** Accelerator **None**, **Internet on**. **Inputs:** this account's dataset `cos30049a2-text150m-base`
(the tokenizer) and notebook `cos30049a2-ds7-data` (DS7 v2). Output `stage/` and `stage_manifest.json`, read by
`cos30049a2-text150m-stages`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this build uses; recorded in stage_manifest.json
PINS = ["numpy==2.2.6", "tokenizers==0.21.1", "huggingface_hub==0.30.2", "pyarrow==20.0.0"]  # environment.yml
GLOSSARY_REPEATS = 4   # the glossary is under 1% of DS7 v2's training rows; repeats give the open questions weight
'''),
    ("code", SETUP + CLONE + FIND),
    ("code", r'''
base = check_base()
TOK = base / "tokenizer.json"
ds7_manifest_path = find_input("data_manifest.json")
ds7_manifest = json.loads(ds7_manifest_path.read_text(encoding="utf-8"))
DS7 = SRC / "data" / "processed" / "ds7_v2"
DS7.mkdir(parents=True, exist_ok=True)
for name in ("train.jsonl", "val.jsonl", "glossary_test.jsonl"):
    rel = f"data/processed/ds7_v2/{name}"
    shutil.copy(ds7_manifest_path.parent / rel, DS7 / name)
    assert sha256(DS7 / name) == ds7_manifest["files"][rel], f"{rel} differs from data_manifest.json"
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

# Stage B3 (docs/TRAINING_PLAN.md): the same data plus the weak-case examples (explorer/weak_cases.py, committed
# with their SHA-256), and the writer's dev files: the weak-case dev slice and DS7 v2's validation split.
WEAK = SRC / "data" / "processed" / "weak_cases"
for f in json.loads((WEAK / "manifest.json").read_text(encoding="utf-8"))["files"]:
    assert sha256(SRC / f["path"]) == f["sha256"], f"{f['path']} differs from its manifest"
sh(sys.executable, "-m", "lm.prepare_chat", "--tokenizer", str(TOK), "--source", "jsonl", "--jsonl", str(DS7 / "train.jsonl"),
   "--val-jsonl", str(DS7 / "val.jsonl"), "--extra", *[str(glossary)] * (GLOSSARY_REPEATS - 1), str(WEAK / "train.jsonl"),
   "--out", str(STAGE / "ds7_v2_weak_chat"), log="prepare-ds7-v2-weak-chat.log")
(STAGE / "dev").mkdir(exist_ok=True)
shutil.copy(WEAK / "dev.jsonl", STAGE / "dev" / "weak_dev.jsonl")
shutil.copy(DS7 / "val.jsonl", STAGE / "dev" / "ds7v2_val.jsonl")
'''),
    ("code", r'''
manifest = {"commit": COMMIT, "ds7_v2_commit": ds7_manifest["commit"],
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {p.relative_to(WORK).as_posix(): sha256(p) for p in sorted(STAGE.rglob("*")) if p.is_file()}}
(WORK / "stage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size / 2**20:9.1f} MB  {p.relative_to(WORK)}")
'''),
]

DEV_EVAL = r'''
# The writer's dev tier (TRAINING_PLAN section 2; settings weakCases): the weak-case dev slice and devVal prompts of
# DS7 v2's validation split. Never the test sets: those are scored once per frozen candidate.
WEAK_SETTINGS = json.loads((SRC / "config" / "settings.json").read_text(encoding="utf-8"))["weakCases"]
DEV = [(STAGE / "dev" / "weak_dev.jsonl", "devweak", 100000), (STAGE / "dev" / "ds7v2_val.jsonl", "devval", WEAK_SETTINGS["devVal"])]


def dev_jobs(model, checkpoint):
    return [(model, checkpoint, test, tag, n, WEAK_SETTINGS["devValSeed"]) for test, tag, n in DEV]


def run_queue(gpu, queue):
    for model, checkpoint, test, tag, n, seed in queue:
        sh(sys.executable, "-m", "lm.eval_explain", "--model", model, "--checkpoint", str(checkpoint), "--test", str(test),
           "--tag", tag, "--n", str(n), "--seed", str(seed), log=f"eval-{model}-{tag}.log",
           env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))


def run_queues(queues):
    threads = [threading.Thread(target=run_queue, args=(gpu, q)) for gpu, q in enumerate(queues)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
'''

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
EVAL_HOURS = 1.25      # what the evaluations need on two GPUs (test sets, then the dev tier)
RESUME_FROM = None     # e.g. "/kaggle/input/<this notebook>/runs" to continue a stopped session
WEAK_CASES = True      # stage B3: the explanation stage adds the weak-case examples; set by the round-1 trial's gate
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
          (config("text_150m_explain_v2", data_dir=STAGE / ("ds7_v2_weak_chat" if WEAK_CASES else "ds7_v2_chat"),
                  out_dir=EXPLAIN, init_from=CHAT / "model.pt"), EXPLAIN)]
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
    ("code", DEV_EVAL + r'''
# Every evaluation the adoption rule needs, one queue per GPU; each test set writes its own tables. --n above every
# set's size: each prompt is answered (the default draws 500). The dev tier comes last (the B3 gate against round 1).
V2 = EXPLAIN / "model_best.pt"
GLOSSARY = STAGE / "glossary_test.jsonl"
RANDOM = SRC / "data" / "processed" / "random_test"
ALL = 100000
FIRST = [("text150m_v2", V2, SRC / "data" / "processed" / "ds7" / "test.jsonl", "", ALL, 30049)] + [
    ("text150m_v2", V2, RANDOM / f"{part}.jsonl", f"random_{part}", ALL, 30049)
    for part in ("real", "hard", "crawl_ds1", "crawl_ds10", "extreme")]
SECOND = [("text150m_v2", V2, SRC / "data" / "processed" / "ds10" / "explain.jsonl", "ds10", ALL, 30049),
          ("text150m_v2", V2, GLOSSARY, "glossary", ALL, 30049),
          ("text150m", base / "text150m_explain_v1.pt", GLOSSARY, "glossary", ALL, 30049)] + [
    ("text150m_v2", V2, RANDOM / f"{part}.jsonl", f"random_{part}", ALL, 30049)
    for part in ("synthetic", "balanced", "crawl_ds2", "close_calls", "max_shift")]
# A test set's evaluations stay in one queue: they write one summary table (so both writers' dev jobs go together).
FIRST += dev_jobs("text150m_v2", V2) + dev_jobs("text150m_r1", base / "text150m_explain_v1.pt")
QUEUES = [FIRST, SECOND] if GPUS > 1 else [FIRST + SECOND]

if (EXPLAIN / "model.pt").exists():
    run_queues(QUEUES)
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


B3_TRIAL_NB = [
    ("markdown", r'''
# COS30049 A2 - Stage B3 tried on the round-1 152M model: does adding weak-case examples help? (T4 x2)

`docs/TRAINING_PLAN.md` stage B3, first tried on round 1 while the round-2 pretraining runs, so the recipe is known
before the round-2 model exists. From round 1's chat model, two explanation stages run side by side, one per GPU,
with `configs/text_150m_explain_v2.json`:

- **A**: DS7 v2 + the glossary x4 (round 2's planned recipe);
- **B**: the same + the weak-case examples (`explorer/weak_cases.py`: other methods, reworded instructions of its
  own, shuffled facts, three or four reasons).

Then round 1's writer, A and B answer the same dev prompts (the weak-case dev slice and a fixed draw of DS7 v2's
validation split). `python -m explorer.cli writer-gate` (run at home on the output tables) decides; the test sets
are not touched here.

**Settings:** Accelerator **GPU T4 x2**, **Internet on**. **Inputs:** this account's datasets
`cos30049a2-text150m-base` and `cos30049a2-text150m-chat-v1` (round 1's chat model) and notebook
`cos30049a2-text150m-stage-data`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses; recorded in kaggle_run.json
SESSION_HOURS = 11.75  # Kaggle ends a GPU session at 12 h
EVAL_HOURS = 1.5       # three writers on two dev files, one queue per GPU
# round 1's chat model (runs/text-150m-chat/model.pt on the PC), uploaded as dataset cos30049a2-text150m-chat-v1
CHAT_V1_SHA256 = "81645aa07bff32e2dc2ee8e94ceafcfeb85a2c0efd8bf2b75d4d826aabc9dfdc"
PINS = ["tokenizers==0.21.1", "numpy==2.2.6"]  # environment.yml; torch is Kaggle's own (recorded)
'''),
    ("code", SETUP + r'''
if shutil.which("nvidia-smi") is None:
    raise SystemExit("No GPU in this session: Settings panel -> Accelerator -> GPU T4 x2 (needs a verified phone).")
GPUS = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
RUNS = WORK / "runs"
''' + CLONE + FIND),
    ("code", r'''
base = check_base()
stage_manifest_path = find_input("stage_manifest.json")
for rel, digest in json.loads(stage_manifest_path.read_text(encoding="utf-8"))["files"].items():
    assert sha256(stage_manifest_path.parent / rel) == digest, f"{rel} differs from stage_manifest.json"
STAGE = stage_manifest_path.parent / "stage"
CHAT_V1 = find_input("text150m_chat_v1.pt")
assert sha256(CHAT_V1) == CHAT_V1_SHA256, "text150m_chat_v1.pt is not round 1's chat model"


def config(name, out, **paths):
    cfg = json.loads((SRC / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    cfg.update({k: str(v) for k, v in paths.items()}, out_dir=str(out), name=out.name)
    path = Path(f"/tmp/{out.name}.json")
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


A, B = RUNS / "text-150m-explain-b3a", RUNS / "text-150m-explain-b3b"
TRIALS = [(config("text_150m_explain_v2", A, data_dir=STAGE / "ds7_v2_chat", init_from=CHAT_V1), A),
          (config("text_150m_explain_v2", B, data_dir=STAGE / "ds7_v2_weak_chat", init_from=CHAT_V1), B)]
'''),
    ("code", r'''
# One stage per GPU at the same time (each single-process), so both see the same session clock.
hours = SESSION_HOURS - (time.time() - T0) / 3600 - EVAL_HOURS - 0.2


def train(gpu, cfg, out):
    sh(sys.executable, "-m", "lm.train", "--config", str(cfg), "--max-hours", f"{hours:.2f}", log=f"{out.name}.log",
       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu % GPUS)))


threads = [threading.Thread(target=train, args=(gpu, cfg, out)) for gpu, (cfg, out) in enumerate(TRIALS)]
for t in threads:
    t.start()
    if GPUS == 1:
        t.join()
for t in threads:
    t.join()
'''),
    ("code", DEV_EVAL + r'''
jobs = dev_jobs("text150m_r1", base / "text150m_explain_v1.pt")
for name, out in (("text150m_b3a", A), ("text150m_b3b", B)):
    if (out / "model.pt").exists():
        jobs += dev_jobs(name, out / "model_best.pt")
# jobs alternate devweak / devval, so each dev file's jobs share one queue (they write one summary table)
run_queues([jobs[0::2], jobs[1::2]] if GPUS > 1 else [jobs])
tables = WORK / "reports" / "tables"
tables.mkdir(parents=True, exist_ok=True)
for f in (SRC / "reports" / "tables").glob("assistant_eval*_dev*.csv"):
    shutil.copy(f, tables)
(WORK / "kaggle_run.json").write_text(json.dumps({
    "commit": COMMIT, "gpus": GPUS, "python": sys.version.split()[0],
    "packages": {p: version(p) for p in ("torch", "numpy", "tokenizers")},
    "session_hours": round((time.time() - T0) / 3600, 2),
    "finished": {out.name: (out / "model.pt").exists() for _, out in TRIALS},
}, indent=2), encoding="utf-8")
'''),
]


DIST_NB = [
    ("markdown", r'''
# COS30049 A2 - Distributions of the scoring methods' recall over many plant seeds (Kaggle CPU session)

Runs `python -m explorer.cli distribution` (`explorer/distribution.py`): the three planted patterns replanted with
100 seeds on DS1 and DS10 and 20 on DS2 (seeds from 70000, disjoint from every tuning and reporting seed), every
frozen method scored on each. Output: `reports/tables/stats_methods_seeds.csv` and `stats_methods_summary.csv`
(mean, spread, 95% interval, paired comparison with the baseline, rank). CPU only, so no GPU quota.

**Settings:** Accelerator **None**, **Internet on**. Then **Save Version -> Save & Run All (Commit)**.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2"]
'''),
    ("code", SETUP + CLONE),
    ("code", r'''
sh(sys.executable, "-m", "explorer.cli", "distribution", log="distribution.log")
tables = WORK / "reports" / "tables"
tables.mkdir(parents=True, exist_ok=True)
for name in ("stats_methods_seeds.csv", "stats_methods_summary.csv"):
    shutil.copy(SRC / "reports" / "tables" / name, tables / name)
shutil.copy(SRC / "reports" / "logs" / "distribution.log", WORK / "distribution_step.log")
(WORK / "kaggle_run.json").write_text(json.dumps({"commit": COMMIT, "hours": round((time.time() - T0) / 3600, 2),
    "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS}}, indent=2), encoding="utf-8")
'''),
]


CLASS_NB = [
    ("markdown", r'''
# COS30049 A2 - Classifier test metrics over training seeds and a bootstrap of the test set (Kaggle CPU session)

Runs `python -m explorer.cli class-distribution` (`explorer/classify_distribution.py`): every reported classifier on
Elliptic (temporal split) and BitcoinHeist (temporal and random schemes), with its tuned settings and validation
threshold, refitted with 10 seeds and its test scores bootstrapped 1,000 times. Elliptic comes from the attached
Kaggle dataset (its licence forbids sharing processed rows); BitcoinHeist is downloaded from UCI. CPU only.

**Settings:** Accelerator **None**, **Internet on**. **Input:** dataset `ellipticco/elliptic-data-set`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2"]
'''),
    ("code", SETUP + CLONE + FIND),
    ("code", r'''
raw = SRC / "data" / "external" / "elliptic"
raw.mkdir(parents=True, exist_ok=True)
for name in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"):
    shutil.copy(find_input(name), raw / name)
sh(sys.executable, "-m", "explorer.cli", "external", "--bitcoinheist", log="external.log")
sh(sys.executable, "-m", "explorer.cli", "process-external", log="process-external.log")
sh(sys.executable, "-m", "explorer.cli", "class-distribution", log="class-distribution.log")
tables = WORK / "reports" / "tables"
tables.mkdir(parents=True, exist_ok=True)
for name in ("stats_classifiers_seeds.csv", "stats_classifiers_summary.csv", "stats_classifiers_paired.csv"):
    shutil.copy(SRC / "reports" / "tables" / name, tables / name)
(WORK / "kaggle_run.json").write_text(json.dumps({"commit": COMMIT, "hours": round((time.time() - T0) / 3600, 2),
    "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS}}, indent=2), encoding="utf-8")
'''),
]


STAGE_CPU_NB = [
    ("markdown", r'''
# COS30049 A2 - One training stage of the DS1 scoring methods, with its dev gate (Kaggle CPU session)

Serves `docs/TRAINING_PLAN.md` (sections 3-5). From the repository at a fixed commit, the DS1 pipeline that fits
every scoring method (`STEPS`) runs twice in this one session: first with `config/settings.json` as committed (the
last accepted stage, `BEFORE`), then with the stage's `OVERRIDES` applied. After each pass, `stage-dev` scores
every method on the **tuning** plants only (seeds 40000-40004; the reporting seeds are never read), and
`stage-compare` applies the gate fixed in advance and writes `reports/tables/stages.csv`. Both passes run on
the same machine and package versions, so the difference is the stage alone. CPU only, so no GPU quota.

**Settings:** Accelerator **None**, **Internet on**. Set `STAGE` and `OVERRIDES` below (or let the push copy set
them), then **Save Version -> Save & Run All (Commit)**.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses
STAGE = "A0"           # the stage's name in stages.csv
BEFORE = "M0"          # the last accepted stage (what settings.json holds at COMMIT)
OVERRIDES = {{"features.pace.enabled": True}}  # dotted settings.json paths -> the stage's values
STEPS = ["features", "features --as-written", "split", "inject", "baseline", "outliers", "clustering", "regression", "fusion"]
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2"]
'''),
    ("code", SETUP + CLONE),
    ("code", r'''
def cli(*args, log):
    sh(sys.executable, "-m", "explorer.cli", *args, log=log)


def run_pass(name):
    for step in STEPS:
        cli(*step.split(), log=f"{name}-{step.replace(' --', '-')}.log")
    cli("stage-dev", "--stage", name, log=f"{name}-stage-dev.log")


def override(settings_path, overrides):
    """Set each dotted path (e.g. features.pace.enabled) in settings.json; every path must exist already."""
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = settings
        for key in parents:
            node = node[key]
        assert leaf in node, f"{dotted} is not in settings.json"
        node[leaf] = value
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


run_pass(BEFORE)
override(SRC / "config" / "settings.json", OVERRIDES)
run_pass(STAGE)
cli("stage-compare", "--before", BEFORE, "--after", STAGE, "--stage", STAGE, log="stage-compare.log")
'''),
    ("code", r'''
# The stage's tables, models and features (to commit if the stage is accepted), its settings, and the logs.
for rel in ["reports/tables", "models", "reports/logs", "data/processed/esplora/features.csv", "config/settings.json"]:
    src = SRC / rel
    dest = WORK / "stage" / rel
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.pt", "assistant"))
    elif src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
(WORK / "kaggle_run.json").write_text(json.dumps({"commit": COMMIT, "stage": STAGE, "before": BEFORE,
    "overrides": OVERRIDES, "steps": STEPS, "hours": round((time.time() - T0) / 3600, 2),
    "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS}}, indent=2), encoding="utf-8")
print((SRC / "reports" / "tables" / "stages.csv").read_text(encoding="utf-8"))
'''),
]


STAGE_SEARCH_NB = [
    ("markdown", r'''
# COS30049 A2 - Training stages C1-C6 and C9 of the DS1 scoring methods, with their dev gate (Kaggle CPU session)

Serves `docs/TRAINING_PLAN.md` (sections 3-4). From the repository at a fixed commit, this rebuilds the DS1 features
and the tuning plants, then `stage-search` searches each scoring method's grid (`config/settings.json ->
stageSearch`) at the method's own M0 flag budget: candidates are ranked on tuning seeds 40000-40002, and the winner is
gated against M0 on tuning seeds 40003-40004 (`explorer/stage_search.py`). The reporting seeds are never read. Each
stage runs as its own step, so one failing stage does not stop the others. CPU only, so no GPU quota.

**Settings:** Accelerator **None**, **Internet on**. Then **Save Version -> Save & Run All (Commit)**. About an hour.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses
STAGES = ["C9"]        # C1-C6 ran 28 Oct (commit 34fdf6c); C9 fuses the methods at their accepted stages
STEPS = ["features", "split", "inject"]   # the data the search reads (M0's dev tables are committed)
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2"]
'''),
    ("code", SETUP + CLONE),
    ("code", r'''
def cli(*args, log):
    sh(sys.executable, "-m", "explorer.cli", *args, log=log)


for step in STEPS:
    cli(*step.split(), log=f"{step}.log")
failed = {}
for stage in STAGES:
    t = time.time()
    try:
        cli("stage-search", "--stages", stage, log=f"stage-search-{stage}.log")
    except subprocess.CalledProcessError as e:
        failed[stage] = str(e)
    print(f"{stage}: {(time.time() - t) / 60:.1f} min{' FAILED' if stage in failed else ''}")
'''),
    ("code", r'''
# The searches, the winners, the dev tables and stages.csv (to commit), the settings and the logs.
for rel in ["reports/tables/stage_search", "reports/tables/stage_dev", "reports/tables/stages.csv", "models/stage",
            "reports/logs", "config/settings.json"]:
    src = SRC / rel
    dest = WORK / "stage" / rel
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
(WORK / "kaggle_run.json").write_text(json.dumps({"commit": COMMIT, "stages": STAGES, "failed": failed,
    "steps": STEPS, "hours": round((time.time() - T0) / 3600, 2),
    "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS}}, indent=2), encoding="utf-8")
print((SRC / "reports" / "tables" / "stages.csv").read_text(encoding="utf-8"))
'''),
]


STAGE_TEST_NB = [
    ("markdown", r"""
# COS30049 A2 - The test tier of the accepted DS1 training stages, and the selection rule (Kaggle CPU session)

Serves `docs/TRAINING_PLAN.md` section 5 ("every accepted candidate once"): with `config/settings.json -> stageModels`
naming each method's accepted stage (C3 LOF, C4 K-Means, C5 DBSCAN, C6 the regression range variant), the DS1
features (with F9, an input of the K-Means winner) and the plants are rebuilt, then every method is scored and
evaluated on the REPORTING plants (seeds 30049-30053) with the existing steps: `scores`, `evaluate` (stability,
sanity and the A1 selection rule: stage C10), `significance` (per-plant hits) and `ds2-check`. `stage-test` saves
this tier's snapshot and gates it against M0's (committed in `reports/tables/stage_test/M0_*`), paired by plant with
an address-level bootstrap. CPU only, so no GPU quota.

**Settings:** Accelerator **None**, **Internet on**. Then **Save Version -> Save & Run All (Commit)**.
"""),
    ("code", f"""
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses
SNAPSHOT = "C"        # this tier's snapshot name (stage_test/C_*.csv)
METHODS = []          # the accepted stages under test (29 Oct: lof, kmeans, dbscan, regression_interval); [] = only
                      # rebuild every table with the methods as they stand (no snapshot, no gate)
STEPS = ["features", "features --as-written", "split", "inject", "scores", "evaluate", "significance", "ds2-check"]
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2"]
"""),
    ("code", SETUP + CLONE),
    ("code", r"""
def cli(*args, log):
    sh(sys.executable, "-m", "explorer.cli", *args, log=log)


for step in STEPS:
    t = time.time()
    cli(*step.split(), log=f"{step.split()[0]}{'-as-written' if '--as-written' in step else ''}.log")
    print(f"{step}: {(time.time() - t) / 60:.1f} min")
if METHODS:
    cli("stage-test", "--snapshot", SNAPSHOT, log="stage-test.log")
    cli("stage-test", "--before", "M0", "--after", SNAPSHOT, "--stage", f"{SNAPSHOT}-test", "--methods", *METHODS,
        log="stage-test.log")
"""),
    ("code", r"""
# Everything the steps rewrote (to commit), and the logs.
for rel in ["reports/tables", "models/selection.json", "data/processed/esplora/features.csv",
            "data/processed/esplora/features_as_written.csv", "data/processed/esplora/scores.csv", "reports/logs"]:
    src = SRC / rel
    dest = WORK / "stage" / rel
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
(WORK / "kaggle_run.json").write_text(json.dumps({"commit": COMMIT, "steps": STEPS, "methods": METHODS,
    "hours": round((time.time() - T0) / 3600, 2),
    "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS}}, indent=2), encoding="utf-8")
print((SRC / "reports" / "tables" / "stages.csv").read_text(encoding="utf-8"))
"""),
]


ENSEMBLE_NB = [
    ("markdown", r'''
# COS30049 A2 - Training stage D6: an ensemble of the Elliptic classifiers, with its gate (Kaggle CPU session)

Runs `python -m explorer.cli ensemble` (`explorer/ensemble.py`, `docs/TRAINING_PLAN.md` section 4): the four reported
all-feature classifiers are fitted on Elliptic's fit steps (1-26); steps 27-30 choose a soft-voting or stacking
ensemble and its threshold; steps 31-34 gate it against the random forest (paired bootstrap of F1, McNemar's test,
no time step falling by more than the random forest's seed spread). Only an accepted ensemble is scored on the test
steps (35-49), once. Elliptic comes from the attached Kaggle dataset (its licence forbids sharing processed rows);
only metrics are written. CPU only.

**Settings:** Accelerator **None**, **Internet on**. **Input:** dataset `ellipticco/elliptic-data-set`.
'''),
    ("code", f'''
REPO = "https://github.com/Defaul119/COS30049A2.git"
COMMIT = "{COMMIT}"   # the code this run uses
PINS = ["numpy==2.2.6", "pandas==2.2.3", "scikit-learn==1.6.1", "scipy==1.17.1", "pyarrow==20.0.0", "joblib==1.4.2"]
'''),
    ("code", SETUP + CLONE + FIND),
    ("code", r'''
raw = SRC / "data" / "external" / "elliptic"
raw.mkdir(parents=True, exist_ok=True)
for name in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"):
    shutil.copy(find_input(name), raw / name)
sh(sys.executable, "-c", "import logging, sys; logging.basicConfig(level=logging.INFO); "
   "from explorer.labelled import run_process_elliptic; sys.exit(run_process_elliptic())", log="process-elliptic.log")
sh(sys.executable, "-m", "explorer.cli", "ensemble", log="ensemble.log")
tables = WORK / "reports" / "tables"
tables.mkdir(parents=True, exist_ok=True)
for name in ("ensemble_search.csv", "ensemble_results.csv", "stages.csv"):
    shutil.copy(SRC / "reports" / "tables" / name, tables / name)
(WORK / "kaggle_run.json").write_text(json.dumps({"commit": COMMIT, "hours": round((time.time() - T0) / 3600, 2),
    "packages": {p.split("==")[0]: version(p.split("==")[0]) for p in PINS}}, indent=2), encoding="utf-8")
print((SRC / "reports" / "tables" / "stages.csv").read_text(encoding="utf-8"))
'''),
]


def metadata(slug, code_file, gpu, kernel_sources, dataset_sources=()):
    return {"id": f"KAGGLE_USERNAME/{slug}", "title": slug, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": list(dataset_sources), "competition_sources": [],
            "kernel_sources": kernel_sources, "model_sources": []}


BASE = ["KAGGLE_USERNAME/cos30049a2-text150m-base"]
for folder, slug, cells, gpu, sources, datasets in [
        ("ds7_data", "cos30049a2-ds7-data", DS7_DATA_NB, False, [], []),
        ("text150m_data", "cos30049a2-text150m-data", TEXT_DATA_NB, False, [], BASE),
        ("text150m_train", "cos30049a2-text150m-train", TEXT_TRAIN_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-text150m-data"], BASE),
        ("text150m_stage_data", "cos30049a2-text150m-stage-data", STAGE_DATA_NB, False,
         ["KAGGLE_USERNAME/cos30049a2-ds7-data"], BASE),
        ("text150m_stages", "cos30049a2-text150m-stages", STAGES_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-text150m-stage-data", "KAGGLE_USERNAME/cos30049a2-text150m-train"], BASE),
        ("text150m_b3_trial", "cos30049a2-text150m-b3-trial", B3_TRIAL_NB, True,
         ["KAGGLE_USERNAME/cos30049a2-text150m-stage-data"], BASE + ["KAGGLE_USERNAME/cos30049a2-text150m-chat-v1"]),
        ("distribution", "cos30049a2-distribution", DIST_NB, False, [], []),
        ("class_distribution", "cos30049a2-class-distribution", CLASS_NB, False, [], ["ellipticco/elliptic-data-set"]),
        ("stage_cpu", "cos30049a2-stage-cpu", STAGE_CPU_NB, False, [], []),
        ("stage_search", "cos30049a2-stage-search", STAGE_SEARCH_NB, False, [], []),
        ("ensemble", "cos30049a2-ensemble", ENSEMBLE_NB, False, [], ["ellipticco/elliptic-data-set"]),
        ("stage_test", "cos30049a2-stage-test", STAGE_TEST_NB, False, [], [])]:
    if ONLY and folder not in ONLY:
        continue
    out = REPO_DIR / "kaggle" / folder
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{folder}.ipynb").write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata(slug, f"{folder}.ipynb", gpu, sources, datasets), indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
