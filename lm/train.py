"""Train a language model from scratch, or fine-tune one of ours.

A JSON config (see configs/) describes the run. To continue after a crash,
power cut or Ctrl+C, run the same command again: it resumes from ckpt_last.pt
in the run folder, with the same data order and learning-rate schedule.

Usage:
    python -m lm.train --config configs/text_150m_pretrain.json --bench   # speed and memory test only
    python -m lm.train --config configs/text_150m_pretrain.json           # train, or resume
    python -m lm.train --config configs/text_150m_chat.json               # chat fine-tuning

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
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch

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
    "eval_batches": 40,
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
    """bf16 on GPUs that support it (RTX 30xx and newer), fp16 on older ones (e.g. Kaggle's T4), fp32 on CPU."""
    if device.type != "cuda":
        return None, contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype, torch.autocast(device_type="cuda", dtype=dtype)


def lr_at(step: int, cfg: dict, max_steps: int) -> float:
    """Linear warm-up, then cosine decay from lr to min_lr."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = min(1.0, (step - cfg["warmup_steps"]) / max(1, max_steps - cfg["warmup_steps"]))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def save_atomic(obj: dict, path: Path) -> None:
    """Write to a temporary file, then rename, so a crash mid-save never corrupts the old file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
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
def evaluate(model, sampler, n_batches, micro_batch, device, autocast) -> float:
    model.eval()
    losses = []
    for i in range(n_batches):
        x, y = sampler.batch(i * micro_batch, micro_batch)
        with autocast:
            _, loss, _ = model(to_device(x, device), to_device(y, device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def bench(model, optimizer, sampler, cfg, device, autocast, seq_len, total_tokens) -> None:
    """Measure speed and memory for several micro-batch sizes, without saving anything."""
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="auto", help="auto, cuda, cpu or mps")
    p.add_argument("--bench", action="store_true", help="measure speed and memory, then exit")
    p.add_argument("--max-steps", type=int, help="stop after this step (testing); the schedule is unchanged")
    p.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype, autocast = precision(device)

    data_dir, out_dir = Path(cfg["data_dir"]), Path(cfg["out_dir"])
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / meta.get("tokenizer", "tokenizer.json")).exists():
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
    optimizer = model.configure_optimizer(cfg["lr"], cfg["weight_decay"], cfg["betas"], device.type)
    if ckpt and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    T, B = model_cfg.seq_len, cfg["micro_batch"]
    if cfg["batch_tokens"] % (B * T):
        raise SystemExit(f"batch_tokens ({cfg['batch_tokens']}) must be a multiple of micro_batch x seq_len ({B * T})")
    seqs_per_step = cfg["batch_tokens"] // T
    accum = seqs_per_step // B
    total_tokens = cfg["total_tokens"] or int(cfg["epochs"] * meta["train_tokens"])
    max_steps = max(1, math.ceil(total_tokens / cfg["batch_tokens"]))
    mask = "train_mask.bin" if (data_dir / "train_mask.bin").exists() else None
    train_data = WindowSampler(data_dir / "train.bin", T, cfg["seed"], mask and data_dir / mask)
    val_data = WindowSampler(data_dir / "val.bin", T, cfg["seed"] + 1,
                             mask and (data_dir / "val_mask.bin"))

    step = ckpt["step"] if ckpt else 0
    best_val = ckpt.get("best_val", float("inf")) if ckpt else float("inf")
    n_params = model.num_params()
    print(f"[train] {cfg.get('name', out_dir.name)} on {device}"
          f"{' (' + torch.cuda.get_device_name(device) + ')' if device.type == 'cuda' else ''}"
          f", {'fp32' if amp_dtype is None else str(amp_dtype).replace('torch.', '')}")
    print(f"[train] model {n_params / 1e6:.1f}M parameters: {model_cfg.n_layer} layers, d_model {model_cfg.d_model}, "
          f"{model_cfg.n_head} heads, context {T}, vocabulary {model_cfg.vocab_size}")
    print(f"[train] data {meta['train_tokens']:,} training tokens; run {total_tokens:,} tokens = {max_steps:,} steps "
          f"of {cfg['batch_tokens']:,} ({B} x {T} x {accum} accumulation)")
    if ckpt:
        print(f"[train] resuming from step {step:,} ({ckpt_path})")
    elif cfg["init_from"]:
        print(f"[train] starting from {cfg['init_from']}")

    run_model = torch.compile(model) if cfg["compile"] else model

    if args.bench:
        bench(run_model, optimizer, train_data, cfg, device, autocast, T, total_tokens)
        return
    if step >= max_steps:
        print(f"[train] already finished ({step:,}/{max_steps:,} steps). Weights: {out_dir / 'model.pt'}")
        return

    def weights(val_loss=None) -> dict:
        return {"model": model.state_dict(), "model_config": model_cfg.to_dict(), "step": step,
                "tokens_seen": step * cfg["batch_tokens"], "val_loss": val_loss,
                "kind": meta.get("kind"), "train_config": cfg}

    def checkpoint() -> None:
        save_atomic({**weights(), "optimizer": optimizer.state_dict(), "best_val": float(best_val)}, ckpt_path)

    log = open(out_dir / "log.jsonl", "a", encoding="utf-8")

    def run_eval() -> float:
        nonlocal best_val
        val = evaluate(run_model, val_data, cfg["eval_batches"], B, device, autocast)
        print(f"[eval] step {step:,} | validation loss {val:.4f} (perplexity {math.exp(min(val, 20)):.1f})", flush=True)
        log.write(json.dumps({"step": step, "val_loss": val, "time": time.time()}) + "\n")
        log.flush()
        if val < best_val:
            best_val = val
            save_atomic(weights(val), out_dir / "model_best.pt")
        return val

    if step == 0:
        run_eval()  # the untrained baseline, about ln(vocabulary size)

    run_model.train()
    flops_per_token = model.flops_per_token()
    last_ckpt, step_times, bad_steps = time.time(), [], 0
    try:
        while step < max_steps:
            if args.max_steps is not None and step >= args.max_steps:
                print(f"[train] stopping at --max-steps {args.max_steps}")
                break
            lr = lr_at(step, cfg, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            t0 = time.time()
            loss_sum = torch.zeros((), device=device)
            for micro in range(accum):
                x, y = train_data.batch(step * seqs_per_step + micro * B, B)
                with autocast:
                    _, loss, _ = run_model(to_device(x, device), to_device(y, device))
                scaler.scale(loss / accum).backward()
                loss_sum += loss.detach()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            train_loss = (loss_sum / accum).item()

            # With fp16, the gradient scaler handles overflowing gradients itself (it skips the
            # step and lowers its scale), so only bf16/fp32 runs check the gradients here.
            grads_bad = not scaler.is_enabled() and not math.isfinite(grad_norm.item())
            if not math.isfinite(train_loss) or grads_bad:
                # Skip a step whose loss or gradients blew up; give up if it keeps happening.
                bad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(f"[train] step {step:,}: non-finite loss or gradients, update skipped ({bad_steps} in a row)")
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
                    mfu = flops_per_token * tok_s / (cfg["peak_tflops"] * 1e12)
                    line += f" | MFU {100 * mfu:.0f}% | {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
                line += f" | ETA {fmt_duration(avg * (max_steps - step))}"
                print(line, flush=True)
                log.write(json.dumps({"step": step, "loss": train_loss, "lr": lr, "grad_norm": grad_norm.item(),
                                      "tok_s": tok_s, "time": time.time()}) + "\n")
                log.flush()

            if step % cfg["eval_every"] == 0 and step < max_steps:
                run_eval()
            if time.time() - last_ckpt > cfg["ckpt_every_minutes"] * 60:
                checkpoint()
                last_ckpt = time.time()
                print(f"[train] checkpoint saved at step {step:,}", flush=True)
    except KeyboardInterrupt:
        print("\n[train] interrupted: saving a checkpoint, please wait ...", flush=True)
        checkpoint()
        print(f"[train] saved at step {step:,}. Run the same command to continue.")
        return
    finally:
        log.close()

    checkpoint()
    if step >= max_steps:
        log = open(out_dir / "log.jsonl", "a", encoding="utf-8")
        val = run_eval()
        log.close()
        save_atomic(weights(val), out_dir / "model.pt")
        print(f"[train] finished: {out_dir / 'model.pt'} (validation loss {val:.4f})")


if __name__ == "__main__":
    main()
