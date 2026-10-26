"""Train a language model from scratch, or fine-tune one of ours.

A JSON config (see configs/) describes the run. To continue after a crash,
power cut or Ctrl+C, run the same command again: it resumes from ckpt_last.pt
in the run folder, with the same data order and learning-rate schedule.

Usage:
    python -m lm.train --config configs/text_150m_pretrain.json --bench   # speed and memory test only
    python -m lm.train --config configs/text_150m_pretrain.json           # train, or resume
    python -m lm.train --config configs/text_150m_chat.json               # chat fine-tuning
    python -m lm.train --config ... --max-hours 11                        # stop and save after 11 h (Kaggle)
    python -m lm.train --config ... --eval-only                           # validation loss of the starting weights
    torchrun --standalone --nproc_per_node 2 -m lm.train --config ...     # both GPUs of a Kaggle T4 x2 session

Round 2 (docs/KAGGLE_LM_PLAN.md, C1-C3) continues the 152M model's pretraining on Kaggle, in sessions of at
most 12 hours on GPUs without bf16: --max-hours ends a session cleanly; under torchrun each GPU takes its
share of every step's micro-batches (lm.dist), which leaves the data order and the steps as on one GPU;
"schedule": "wsd" holds the learning rate constant and decays it only over the last `decay_fraction` of
the tokens, so `total_tokens` can still be raised between sessions until the decay begins; and
"eval_windows" fixes the validation windows, so a loss stays comparable with round 1's at any micro-batch.

Files written to the run folder (out_dir in the config):
    ckpt_last.pt     everything needed to resume (model, optimizer, step)
    model.pt         final weights: use for chatting or as the next stage's init_from
    model_best.pt    weights with the lowest validation loss so far
    tokenizer.json   copy of the tokenizer, so the folder is self-contained
    log.jsonl        one line per step and per evaluation, for plots in the report
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from . import dist
from .data import WindowSampler
from .model import GPT, GPTConfig, pad_vocab

DEFAULTS = {
    "seed": 1337,
    "init_from": None,          # weights to start from (chat fine-tuning starts from the pretrained model)
    "total_tokens": None,       # tokens to train on; or set "epochs" instead
    "epochs": None,
    "batch_tokens": 262144,     # tokens per optimizer step (micro_batch x seq_len x accumulation steps)
    "micro_batch": 8,           # sequences per forward pass; lower it if the GPU runs out of memory
    "lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
    "grad_clip": 1.0,
    "eval_every": 200,          # steps between validation runs
    "early_stop_evals": None,   # stop after this many evaluations without a new best (None: never)
    "eval_batches": 40,
    "eval_windows": None,       # validation windows per evaluation (None: eval_batches x micro_batch)
    "schedule": "cosine",       # or "wsd": warm-up, constant, then linear decay over the last decay_fraction
    "decay_fraction": 0.15,     # wsd: share of total_tokens spent decaying to min_lr
    "log_every": 1,
    "ckpt_every_minutes": 30,
    "compile": False,           # torch.compile: faster, but needs Triton (Linux/WSL2)
    "peak_tflops": 25.4,        # RTX 3060 bf16 peak; only used for the efficiency (MFU) figure
    "bench_micro_batches": [16, 8, 4, 2],
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precision(device: torch.device):
    """bf16 on GPUs with native bf16 (RTX 30xx and newer), fp16 on older ones (e.g. Kaggle's T4), fp32 on CPU.

    Compute capability, not torch.cuda.is_bf16_supported(), which recent PyTorch also answers yes on a T4
    because it can emulate bf16 (several times slower).
    """
    if device.type != "cuda":
        return None, contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return dtype, torch.autocast(device_type="cuda", dtype=dtype)


def lr_at(step: int, cfg: dict, max_steps: int) -> float:
    """Linear warm-up, then cosine decay from lr to min_lr; or, with "schedule": "wsd", constant until the
    last decay_fraction of the steps and a linear decay to min_lr over them (warm-up, stable, decay)."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    if cfg.get("schedule", "cosine") == "wsd":
        decay_start = max(cfg["warmup_steps"], round(max_steps * (1 - cfg["decay_fraction"])))
        progress = min(1.0, max(0, step - decay_start) / max(1, max_steps - decay_start))
        return cfg["lr"] + (cfg["min_lr"] - cfg["lr"]) * progress
    progress = min(1.0, (step - cfg["warmup_steps"]) / max(1, max_steps - cfg["warmup_steps"]))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def save_atomic(obj: dict, path: Path) -> None:
    """Write to a temporary file, flush it to disk, then rename, so a crash mid-save never corrupts the old file.

    The fsync matters for power cuts (24 Oct): without it the rename can reach the disk before the data does,
    leaving a renamed file that is empty or zero-filled where the old checkpoint used to be.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    return f"{h}h {m}m" if h else (f"{m}m" if m else f"{s}s")


@torch.no_grad()
def evaluate(model, sampler, n_windows, micro_batch, device, autocast, d: dist.Dist = dist.Dist()) -> float:
    """Mean loss over the validation stream's first n_windows windows (the same ones at any micro-batch),
    shared between the processes."""
    model.eval()
    total, count = 0.0, 0
    for start in range(d.rank * micro_batch, n_windows, d.world * micro_batch):
        n = min(micro_batch, n_windows - start)
        x, y = sampler.batch(start, n)
        with autocast:
            _, loss, _ = model(to_device(x, device), to_device(y, device))
        total += loss.item() * n
        count += n
    model.train()
    total, count = dist.total(d, total, count, device=device)
    return total / count


def bench(model, optimizer, sampler, cfg, device, autocast, seq_len, total_tokens) -> tuple[int, float] | None:
    """Measure speed and memory for several micro-batch sizes, without saving anything.

    Returns (micro_batch, tokens/s) of the fastest size that fits, or None if none fits.
    """
    print("\n[bench] micro_batch | tokens/s | peak GPU memory | time for the full run")
    sizes = cfg["bench_micro_batches"] if device.type == "cuda" else [cfg["micro_batch"]]
    vram = torch.cuda.get_device_properties(device).total_memory / 1e9 if device.type == "cuda" else None
    best = None
    for mb in sizes:
        if cfg["batch_tokens"] % (mb * seq_len):
            continue
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            timed, t0 = 0, None
            for i in range(13):
                if i == 3:  # three warm-up passes (the first also creates the optimizer state)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                x, y = sampler.batch(i * mb, mb)
                with autocast:
                    _, loss, _ = model(to_device(x, device), to_device(y, device))
                loss.backward()
                if i == 0:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if i >= 3:
                    timed += mb * seq_len
            if device.type == "cuda":
                torch.cuda.synchronize()
            tok_s = timed / (time.time() - t0)
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                mem_text = f"{mem:>6.1f} GB of {vram:.1f}"
            else:
                mem, mem_text = 0.0, "   n/a (CPU)"
            print(f"[bench] {mb:>11} | {tok_s:>8,.0f} | {mem_text} | {fmt_duration(total_tokens / tok_s)}")
            # Leave headroom: the desktop and other programs also use the GPU.
            if (vram is None or mem < 0.9 * vram) and (best is None or tok_s > best[1]):
                best = (mb, tok_s)
            out_of_memory = False
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True
        # Free memory only after leaving the except block: until then the exception's traceback keeps
        # the failed step's activations alive, and every smaller size would run out of memory too
        # (on Windows, spilling into system RAM and running ~8x slower instead of failing).
        if out_of_memory:
            print(f"[bench] {mb:>11} | out of memory")
            x = y = loss = None
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if best:
        print(f"\n[bench] fastest that fits: \"micro_batch\": {best[0]} "
              f"(about {fmt_duration(total_tokens / best[1])} for {total_tokens:,} tokens)")
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="auto", help="auto, cuda, cpu or mps")
    p.add_argument("--bench", action="store_true", help="measure speed and memory, then exit")
    p.add_argument("--eval-only", action="store_true",
                   help="print the starting weights' validation loss (resumed, init_from or random), then exit")
    p.add_argument("--max-steps", type=int, help="stop after this step (testing); the schedule is unchanged")
    p.add_argument("--max-hours", type=float, metavar="H",
                   help="save and stop after H hours of training; the same command resumes (Kaggle's 12-hour sessions)")
    p.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    p.add_argument("--seed", type=int, help="override the config's seed; the run folder gets a -s<seed> suffix")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        # One folder per seed, so repeated runs of one config (TxGPT: 3 seeds per size) never share a checkpoint.
        cfg["seed"] = args.seed
        cfg["out_dir"] = f"{cfg['out_dir']}-s{args.seed}"
    device = pick_device(args.device)
    d = dist.setup(device.type)
    if d.on and device.type == "cuda":
        device = torch.device("cuda", d.local_rank)
    say = print if d.main else (lambda *a, **k: None)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype, autocast = precision(device)

    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["out_dir"])
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if d.main and (data_dir / meta.get("tokenizer", "tokenizer.json")).exists():
        shutil.copyfile(data_dir / meta.get("tokenizer", "tokenizer.json"), out_dir / "tokenizer.json")

    # Where do the weights come from: our own checkpoint (resume), an earlier stage, or random?
    ckpt_path = out_dir / "ckpt_last.pt"
    ckpt = None
    if ckpt_path.exists() and not args.fresh:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_cfg = GPTConfig(**ckpt["model_config"])
    elif cfg["init_from"]:
        base = torch.load(cfg["init_from"], map_location="cpu", weights_only=True)
        model_cfg = GPTConfig(**base["model_config"])
    else:
        model_cfg = GPTConfig(**{**cfg["model"], "vocab_size": pad_vocab(meta["vocab_size"])})
    if meta["vocab_size"] > model_cfg.vocab_size:
        raise SystemExit(f"data uses {meta['vocab_size']} token ids but the model has only {model_cfg.vocab_size}")

    model = GPT(model_cfg)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    elif cfg["init_from"]:
        model.load_state_dict(base["model"])
        del base
    model.to(device)
    dist.broadcast_parameters(d, list(model.parameters()))
    optimizer = model.configure_optimizer(cfg["lr"], cfg["weight_decay"], cfg["betas"], device.type)
    if ckpt and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    T, B = model_cfg.seq_len, cfg["micro_batch"]
    if cfg["batch_tokens"] % (B * T * d.world):
        raise SystemExit(f"batch_tokens ({cfg['batch_tokens']}) must be a multiple of micro_batch x seq_len x "
                         f"processes ({B * T * d.world})")
    seqs_per_step = cfg["batch_tokens"] // T
    accum = seqs_per_step // B
    local_accum = accum // d.world  # this process's share of each step's micro-batches
    total_tokens = cfg["total_tokens"] or int(cfg["epochs"] * meta["train_tokens"])
    max_steps = max(1, math.ceil(total_tokens / cfg["batch_tokens"]))
    eval_windows = cfg["eval_windows"] or cfg["eval_batches"] * B
    mask = "train_mask.bin" if (data_dir / "train_mask.bin").exists() else None
    train_data = WindowSampler(data_dir / "train.bin", T, cfg["seed"], mask and data_dir / mask)
    val_data = WindowSampler(data_dir / "val.bin", T, cfg["seed"] + 1,
                             mask and (data_dir / "val_mask.bin"))

    step = ckpt["step"] if ckpt else 0
    best_val = ckpt.get("best_val", float("inf")) if ckpt else float("inf")
    evals_since_best = ckpt.get("evals_since_best", 0) if ckpt else 0
    stopped_early = bool(ckpt.get("stopped_early")) if ckpt else False
    n_params = model.num_params()
    say(f"[train] {cfg.get('name', out_dir.name)} on {d.world} x {device.type}"
        f"{' (' + torch.cuda.get_device_name(device) + ')' if device.type == 'cuda' else ''}"
        f", {'fp32' if amp_dtype is None else str(amp_dtype).replace('torch.', '')}")
    say(f"[train] model {n_params / 1e6:.1f}M parameters: {model_cfg.n_layer} layers, d_model {model_cfg.d_model}, "
        f"{model_cfg.n_head} heads, context {T}, vocabulary {model_cfg.vocab_size}")
    say(f"[train] data {meta['train_tokens']:,} training tokens; run {total_tokens:,} tokens = {max_steps:,} steps "
        f"of {cfg['batch_tokens']:,} ({B} x {T} x {accum} accumulation{f' over {d.world} processes' if d.on else ''})"
        f"; {cfg['schedule']} schedule")
    if ckpt:
        say(f"[train] resuming from step {step:,} ({ckpt_path})")
    elif cfg["init_from"]:
        say(f"[train] starting from {cfg['init_from']}")

    run_model = torch.compile(model) if cfg["compile"] else model

    if args.eval_only:
        val = evaluate(run_model, val_data, eval_windows, B, device, autocast, d)
        say(f"[eval] validation loss {val:.4f} on {eval_windows} windows (perplexity {math.exp(min(val, 20)):.1f})")
        if d.main:
            (out_dir / "eval_only.json").write_text(json.dumps(
                {"val_loss": val, "windows": eval_windows, "step": step, "precision": str(amp_dtype),
                 "weights": str(ckpt_path if ckpt else cfg["init_from"])}, indent=2) + "\n", encoding="utf-8")
        dist.cleanup(d)
        return
    if args.bench:
        # Each process measures its own GPU; the gradient exchange adds well under 1% at these step sizes.
        with contextlib.redirect_stdout(sys.stdout if d.main else io.StringIO()):
            best = bench(run_model, optimizer, train_data, cfg, device, autocast, T, total_tokens // d.world)
        if d.main and best:
            (out_dir / "bench.json").write_text(json.dumps(
                {"micro_batch": best[0], "tok_per_s_per_gpu": best[1], "gpus": d.world, "compile": cfg["compile"],
                 "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type}, indent=2)
                + "\n", encoding="utf-8")
        dist.cleanup(d)
        return
    if step >= max_steps or stopped_early:
        say(f"[train] already finished ({step:,}/{max_steps:,} steps"
            f"{', stopped early' if stopped_early else ''}). Weights: {out_dir / 'model.pt'}")
        dist.cleanup(d)
        return

    def weights(val_loss=None) -> dict:
        return {"model": model.state_dict(), "model_config": model_cfg.to_dict(), "step": step,
                "tokens_seen": step * cfg["batch_tokens"], "val_loss": val_loss,
                "kind": meta.get("kind"), "train_config": cfg}

    def checkpoint() -> None:
        if d.main:
            save_atomic({**weights(), "optimizer": optimizer.state_dict(), "best_val": float(best_val),
                         "evals_since_best": evals_since_best, "stopped_early": stopped_early}, ckpt_path)

    log = open(out_dir / "log.jsonl", "a", encoding="utf-8") if d.main else None

    def run_eval() -> float:
        nonlocal best_val, evals_since_best
        val = evaluate(run_model, val_data, eval_windows, B, device, autocast, d)
        say(f"[eval] step {step:,} | validation loss {val:.4f} (perplexity {math.exp(min(val, 20)):.1f})", flush=True)
        if log:
            log.write(json.dumps({"step": step, "val_loss": val, "time": time.time()}) + "\n")
            log.flush()
        # Every process computed the same mean, so they agree on the best step and on stopping early.
        if val < best_val:
            best_val, evals_since_best = val, 0
            if d.main:
                save_atomic(weights(val), out_dir / "model_best.pt")
        else:
            evals_since_best += 1
        return val

    if step == 0:
        run_eval()  # the untrained baseline, about ln(vocabulary size)

    run_model.train()
    flops_per_token = model.flops_per_token()
    started, last_ckpt, step_times, bad_steps = time.time(), time.time(), [], 0
    out_of_time = False
    try:
        while step < max_steps:
            if args.max_steps is not None and step >= args.max_steps:
                say(f"[train] stopping at --max-steps {args.max_steps}")
                break
            lr = lr_at(step, cfg, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            t0 = time.time()
            loss_sum = torch.zeros((), device=device)
            # Micro-batch m of this step is the stream's position step x seqs_per_step + m x B on any number of
            # processes; process r takes m = r, r + world, ... and lm.dist averages the processes' gradients.
            for micro in range(d.rank, accum, d.world):
                x, y = train_data.batch(step * seqs_per_step + micro * B, B)
                with autocast:
                    _, loss, _ = run_model(to_device(x, device), to_device(y, device))
                scaler.scale(loss / local_accum).backward()
                loss_sum += loss.detach()
            dist.average_gradients(d, list(model.parameters()))
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            (loss_total,) = dist.total(d, (loss_sum / local_accum).item(), device=device)
            train_loss = loss_total / d.world

            # With fp16, the gradient scaler handles overflowing gradients itself (it skips the
            # step and lowers its scale), so only bf16/fp32 runs check the gradients here.
            grads_bad = not scaler.is_enabled() and not math.isfinite(grad_norm.item())
            if not math.isfinite(train_loss) or grads_bad:
                # Skip a step whose loss or gradients blew up; give up if it keeps happening.
                bad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                say(f"[train] step {step:,}: non-finite loss or gradients, update skipped ({bad_steps} in a row)")
                if bad_steps >= 5:
                    raise SystemExit("[train] training diverged; resume from the last checkpoint with a lower lr")
                step += 1
                continue
            bad_steps = 0
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            step_times.append(time.time() - t0)
            step_times = step_times[-20:]
            if step % cfg["log_every"] == 0 or step == 1:
                avg = sum(step_times) / len(step_times)
                tok_s = cfg["batch_tokens"] / avg
                line = (f"[train] step {step:,}/{max_steps:,} | loss {train_loss:.4f} | lr {lr:.2e} | "
                        f"grad {grad_norm.item():.2f} | {tok_s:,.0f} tok/s")
                if device.type == "cuda":
                    mfu = flops_per_token * tok_s / (cfg["peak_tflops"] * 1e12 * d.world)
                    line += f" | MFU {100 * mfu:.0f}% | {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
                line += f" | ETA {fmt_duration(avg * (max_steps - step))}"
                say(line, flush=True)
                if log:
                    log.write(json.dumps({"step": step, "loss": train_loss, "lr": lr, "grad_norm": grad_norm.item(),
                                          "tok_s": tok_s, "time": time.time()}) + "\n")
                    log.flush()

            if step % cfg["eval_every"] == 0 and step < max_steps:
                run_eval()
                if cfg["early_stop_evals"] and evals_since_best >= cfg["early_stop_evals"]:
                    stopped_early = True
                    say(f"[train] no improvement in {evals_since_best} evaluations: stopping early "
                        f"(best validation loss {best_val:.4f} in model_best.pt)", flush=True)
                    break
            if time.time() - last_ckpt > cfg["ckpt_every_minutes"] * 60:
                checkpoint()
                last_ckpt = time.time()
                say(f"[train] checkpoint saved at step {step:,}", flush=True)
            if args.max_hours and dist.any_true(d, time.time() - started > args.max_hours * 3600, device):
                out_of_time = True
                break
    except KeyboardInterrupt:
        say("\n[train] interrupted: saving a checkpoint, please wait ...", flush=True)
        checkpoint()
        say(f"[train] saved at step {step:,}. Run the same command to continue.")
        return
    finally:
        if log:
            log.close()

    checkpoint()
    if out_of_time and step < max_steps:
        say(f"[train] --max-hours {args.max_hours} reached: saved at step {step:,}. Run the same command to continue.")
    elif step >= max_steps or stopped_early:
        log = open(out_dir / "log.jsonl", "a", encoding="utf-8") if d.main else None
        val = run_eval()
        if log:
            log.close()
        if d.main:
            save_atomic(weights(val), out_dir / "model.pt")
        say(f"[train] finished: {out_dir / 'model.pt'} (validation loss {val:.4f})")
    dist.barrier(d)
    dist.cleanup(d)


if __name__ == "__main__":
    main()
