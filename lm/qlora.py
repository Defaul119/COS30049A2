"""QLoRA fine-tuning of a pretrained ~3B model: the Explorer Assistant 3B.

Serves plan §9 (docs/plan/sections/s09-lm.tex, "The Explorer Assistant 3B"), an extra model added
alongside the two trained from scratch. The base weights load in 4-bit NF4 (bitsandbytes) and only
low-rank adapters train, which fits a 12 GB RTX 3060. The base is Qwen2.5-3B-Instruct, which is
already chat-tuned, so one stage is enough (configs/qwen3b_explain.json): DS7 mixed with a small
DS6 (smol-smoltalk) subset, which keeps its general chat from degrading while it learns the
explanations. Several train files are simply concatenated and shuffled together.

Loss is on assistant replies only: every other token's label is -100. Conversations are rendered
with the base model's own chat template; one longer than seq_len is cut after the last reply that
fits, as lm.prepare_chat does. The DS7 splits are the ones the 152M model uses.

Like lm.train, a run resumes: Ctrl+C (or a crash) keeps the latest checkpoint, and the same command
continues with the same data order and learning-rate schedule. The GPU must be free (pretraining
and TxGPT have priority; plan §9 "When"). It also runs on Kaggle (kaggle/qwen3b_explain.ipynb): on GPUs
without native bf16 (T4, P100) it computes in fp16 with loss scaling, and --max-hours stops it cleanly
inside a session's time limit.

Usage (cos30049-a2-gpu environment):
    python -m lm.qlora --config configs/qwen3b_explain.json
    python -m lm.qlora --config configs/qwen3b_explain.json --bench   # 5 steps on the longest batches: speed, memory
    python -m lm.qlora --config configs/qwen3b_explain.json --smoke   # tiny random model, CPU, a few steps
    python -m lm.qlora --config configs/qwen3b_explain.json --max-hours 11   # stop and save after 11 h

Files written to the run folder (out_dir):
    adapter/           latest LoRA adapter (resume point; or a later stage's init_adapter)
    adapter_best/      adapter with the lowest validation loss: the one to evaluate and ship
    state.pt           adapter weights, optimizer, step and data order, for resuming
    log.jsonl          one line per step and per evaluation
    meta.json          base model and revision, data files with SHA-256, counts, speed and memory

`retention_val_files` (optional) are general-chat conversations held out from training. Their loss is
logged at every evaluation next to the main validation loss, as evidence that fine-tuning on DS7 did
not degrade the model's general chat; the best adapter is still chosen on `val_files` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from .train import fmt_duration, lr_at, save_atomic

DEFAULTS: dict[str, Any] = {
    "seed": 1,
    "revision": None,            # a commit hash of the base model's Hugging Face repository (recorded)
    "init_adapter": None,        # a previous stage's adapter folder to continue training
    "seq_len": 1024,
    "micro_batch": 4,
    "grad_accum": 4,             # micro-batches per optimizer step
    "epochs": 1,
    "max_steps": None,           # overrides epochs when set
    "lr": 2e-4,
    "min_lr": 2e-5,
    "warmup_steps": 30,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "eval_every": 100,
    "eval_batches": 40,
    "ckpt_every_minutes": 20,
    "load_in_4bit": True,
    "retention_val_files": [],   # general-chat conversations whose loss is only logged (see the module docstring)
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
}
IGNORE = -100  # the label PyTorch's cross-entropy skips


def load_config(path: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


# -- data ------------------------------------------------------------------------------------


def read_conversations(paths: list[str]) -> list[list[dict[str, str]]]:
    """Messages lists from JSON Lines files ({"messages": [...]} per line, as DS7 and the chat subset)."""
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            out += [json.loads(line)["messages"] for line in fh if line.strip()]
    return out


def encode(tok, messages: list[dict[str, str]], seq_len: int) -> tuple[list[int], list[int], bool] | None:
    """(input ids, labels, truncated) for one conversation, labels -100 except on assistant replies.

    Each reply's span is found by rendering the chat template up to the reply's generation prompt
    and up to the end of the reply; the template must render each conversation as a prefix of
    the next, which ChatML templates (Qwen) do and which is checked here. Returns None when even
    the first reply does not fit in seq_len.
    """
    ids: list[int] = []
    labels: list[int] = []
    ends: list[int] = []
    done = ""
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        head = tok.apply_chat_template(messages[:k], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(messages[: k + 1], tokenize=False)
        if not (head.startswith(done) and full.startswith(head)):
            raise ValueError("the chat template does not render conversations as prefixes; spans cannot be found")
        for text, learn in ((head[len(done):], False), (full[len(head):], True)):
            piece = tok(text, add_special_tokens=False)["input_ids"]
            ids += piece
            labels += piece if learn else [IGNORE] * len(piece)
        ends.append(len(ids))
        done = full
    fitting = [e for e in ends if e <= seq_len]
    if not fitting:
        return None
    return ids[: fitting[-1]], labels[: fitting[-1]], len(fitting) < len(ends)


def encode_all(tok, conversations: list[list[dict[str, str]]], seq_len: int) -> tuple[list[tuple[list[int], list[int]]], dict[str, int]]:
    examples, stats = [], {"conversations": len(conversations), "too_long": 0, "truncated": 0}
    for messages in conversations:
        enc = encode(tok, messages, seq_len)
        if enc is None:
            stats["too_long"] += 1
            continue
        ids, labels, truncated = enc
        stats["truncated"] += truncated
        examples.append((ids, labels))
    stats["examples"] = len(examples)
    stats["tokens"] = sum(len(ids) for ids, _ in examples)
    stats["reply_tokens"] = sum(sum(l != IGNORE for l in labels) for _, labels in examples)
    return examples, stats


def batches(examples: list, micro_batch: int, rng: random.Random) -> list[list[int]]:
    """One epoch of micro-batches (lists of example indices): shuffled, then grouped by length within
    windows of 50 micro-batches so a batch pads to similar lengths without fixing the order."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    window = micro_batch * 50
    out = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start:start + window], key=lambda i: len(examples[i][0]))
        out += [chunk[j:j + micro_batch] for j in range(0, len(chunk), micro_batch)]
    rng.shuffle(out)
    return out


def collate(examples: list, idx: list[int], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(examples[i][0]) for i in idx)
    ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for row, i in enumerate(idx):
        x, y = examples[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# -- model -----------------------------------------------------------------------------------


def load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def compute_dtype() -> torch.dtype:
    """bf16 on GPUs with native bf16 (Ampere and newer, e.g. the RTX 3060), fp16 on older ones (Kaggle's T4, P100).

    Compute capability, not torch.cuda.is_bf16_supported(): recent PyTorch reports bf16 as supported on a T4
    because it can emulate it, which is several times slower.
    """
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_model(cfg: dict[str, Any], smoke: bool):
    """The base model (4-bit on the GPU, or a tiny random one for --smoke) with trainable LoRA adapters."""
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    if smoke:
        # Same architecture and vocabulary as the base model, a few thousand times smaller.
        small = AutoConfig.from_pretrained(cfg["base_model"], revision=cfg["revision"])
        small.update({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                      "num_attention_heads": 4, "num_key_value_heads": 2})
        torch.manual_seed(cfg["seed"])
        model = AutoModelForCausalLM.from_config(small)
    else:
        dtype = compute_dtype()
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dtype) if cfg["load_in_4bit"] else None
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["revision"],
                                                     quantization_config=quant, torch_dtype=dtype,
                                                     device_map={"": 0})
        # Gradient checkpointing trades about a third more compute for activations that fit in 12 GB.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if cfg["init_adapter"]:
        return PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True)
    lora = cfg["lora"]
    return get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                            target_modules=lora["target_modules"], task_type="CAUSAL_LM"))


@torch.no_grad()
def evaluate(model, examples: list, val_batches: list[list[int]], pad_id: int, device: torch.device) -> float:
    """Mean loss per reply token over the fixed validation micro-batches."""
    model.eval()
    total, count = 0.0, 0
    for idx in val_batches:
        batch = {k: v.to(device) for k, v in collate(examples, idx, pad_id).items()}
        n = int((batch["labels"][:, 1:] != IGNORE).sum())
        total += model(**batch).loss.item() * n
        count += n
    model.train()
    return total / max(count, 1)


def bench(model, optimizer, scaler, params: list, examples: list, cfg: dict[str, Any], pad_id: int,
          device: torch.device, steps: int) -> None:
    """Optimizer steps on the longest examples first, so an out-of-memory error shows in minutes, not hours.

    Prints tokens per second and peak GPU memory; nothing is saved. The longest batches are the worst
    case for memory, so real training (shorter batches on average) runs a little faster.
    """
    longest = sorted(range(len(examples)), key=lambda i: -len(examples[i][0]))
    micro = [longest[j:j + cfg["micro_batch"]] for j in range(0, len(longest), cfg["micro_batch"])]
    needed = steps * cfg["grad_accum"]
    micro = (micro * (needed // max(len(micro), 1) + 1))[:needed]
    tokens, started = 0, time.time()
    for s in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for idx in micro[s * cfg["grad_accum"]:(s + 1) * cfg["grad_accum"]]:
            batch = {k: v.to(device) for k, v in collate(examples, idx, pad_id).items()}
            scaler.scale(model(**batch).loss / cfg["grad_accum"]).backward()
            tokens += int(batch["attention_mask"].sum())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        memory = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        print(f"[bench] step {s + 1}/{steps}: {tokens / (time.time() - started):,.0f} tok/s, peak memory {memory:.1f} GB",
              flush=True)
    print("[bench] done; nothing saved. If peak memory is near the GPU's limit, lower micro_batch and raise grad_accum.")


def save_adapter(model, path: Path) -> None:
    """save_pretrained(path), but a crash or power cut mid-save leaves the previous adapter whole (24 Oct).

    The adapter is written to a scratch folder and each file is flushed to disk, then moved over the old
    one; each move is atomic. adapter_config.json is the same at every save, so a folder caught between
    two moves is still a consistent adapter.
    """
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    path.mkdir(parents=True, exist_ok=True)
    for f in sorted(tmp.iterdir()):
        with open(f, "r+b") as fh:  # Windows only flushes a file opened for writing
            os.fsync(fh.fileno())
        os.replace(f, path / f.name)
    tmp.rmdir()


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true", help="tiny random model on the CPU for a few steps (tests the code path)")
    p.add_argument("--bench", type=int, nargs="?", const=5, metavar="STEPS",
                   help="run STEPS optimizer steps on the longest batches, print speed and peak memory, save nothing")
    p.add_argument("--max-hours", type=float, metavar="H",
                   help="stop cleanly (saving) after H hours of training; the same command resumes. For time-limited "
                        "sessions such as Kaggle's 12-hour limit")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.update({"out_dir": cfg["out_dir"] + "-smoke", "max_steps": 3, "micro_batch": 2, "grad_accum": 1,
                    "eval_every": 2, "eval_batches": 2, "init_adapter": None, "seq_len": min(cfg["seq_len"], 512)})
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.smoke else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU: run with --smoke to test on the CPU, or use the cos30049-a2-gpu environment")

    tok = load_tokenizer(cfg)
    train_conv, val_conv = read_conversations(cfg["train_files"]), read_conversations(cfg["val_files"])
    if args.smoke:
        train_conv, val_conv = train_conv[:16], val_conv[:8]
    train, train_stats = encode_all(tok, train_conv, cfg["seq_len"])
    val, val_stats = encode_all(tok, val_conv, cfg["seq_len"])
    retention, retention_stats = encode_all(tok, read_conversations(cfg["retention_val_files"]), cfg["seq_len"])
    print(f"[qlora] train {train_stats}\n[qlora] val {val_stats}\n[qlora] retention {retention_stats}", flush=True)

    rng = random.Random(cfg["seed"])
    steps_per_epoch = math.ceil(math.ceil(len(train) / cfg["micro_batch"]) / cfg["grad_accum"])
    max_steps = cfg["max_steps"] or cfg["epochs"] * steps_per_epoch
    val_batches = batches(val, cfg["micro_batch"], random.Random(cfg["seed"]))[: cfg["eval_batches"]]
    retention_batches = batches(retention, cfg["micro_batch"], random.Random(cfg["seed"]))[: cfg["eval_batches"]]

    model = load_model(cfg, args.smoke)
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # fp16 needs loss scaling (small gradients underflow to zero); with bf16 or on the CPU the scaler does nothing.
    fp16 = device.type == "cuda" and compute_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    n_trainable = sum(q.numel() for q in params)
    hardware = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    print(f"[qlora] {cfg['base_model']} @ {cfg['revision']}: {n_trainable:,} trainable parameters; "
          f"{max_steps} steps of {cfg['micro_batch']}x{cfg['grad_accum']} sequences; {hardware}, "
          f"{'fp16 with loss scaling' if fp16 else 'bf16' if device.type == 'cuda' else 'fp32'}", flush=True)

    if args.bench:
        bench(model, optimizer, scaler, params, train, cfg, tok.pad_token_id, device, args.bench)
        return

    # Resume: the adapter weights, optimizer, step and the epoch's batch order.
    step, epoch, cursor, best = 0, 0, 0, float("inf")
    epoch_batches = batches(train, cfg["micro_batch"], rng)
    state_path = out / "state.pt"
    if state_path.exists() and not args.smoke:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = torch.load(state_path, weights_only=False)
        # state.pt carries its own copy of the adapter (one atomic file), so the weights always match the step;
        # older state files without it fall back to the adapter folder.
        set_peft_model_state_dict(model, state["adapter"] if "adapter" in state
                                  else load_file(str(out / "adapter" / "adapter_model.safetensors")))
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        step, epoch, cursor, best = state["step"], state["epoch"], state["cursor"], state["best"]
        rng.setstate(state["rng"])
        epoch_batches = state["epoch_batches"]
        print(f"[qlora] resumed at step {step} (epoch {epoch}, batch {cursor}); best validation loss {best:.4f}", flush=True)

    def save(tag: str = "adapter") -> None:
        from peft import get_peft_model_state_dict

        save_adapter(model, out / tag)
        if tag == "adapter":
            save_atomic({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "cursor": cursor, "best": best,
                         "rng": rng.getstate(), "epoch_batches": epoch_batches}, state_path)

    meta = {"config": cfg, "smoke": args.smoke, "trainable_parameters": n_trainable, "train": train_stats,
            "val": val_stats, "retention": retention_stats,
            "data": [{"path": f, "sha256": sha256(f)}
                     for f in cfg["train_files"] + cfg["val_files"] + cfg["retention_val_files"]]}
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    started, last_ckpt, tokens_seen, first_step = time.time(), time.time(), 0, step
    try:
        while step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(cfg["grad_accum"]):
                if cursor >= len(epoch_batches):
                    epoch, cursor = epoch + 1, 0
                    epoch_batches = batches(train, cfg["micro_batch"], rng)
                batch = {k: v.to(device) for k, v in collate(train, epoch_batches[cursor], tok.pad_token_id).items()}
                cursor += 1
                loss = model(**batch).loss / cfg["grad_accum"]
                scaler.scale(loss).backward()
                loss_sum += loss.item()
                tokens_seen += int(batch["attention_mask"].sum())
            lr = lr_at(step, cfg, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)  # so the clipping threshold applies to the true gradients
            grad = float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            elapsed = time.time() - started
            rate = tokens_seen / max(elapsed, 1e-9)
            memory = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            print(f"[qlora] step {step:,}/{max_steps:,} | loss {loss_sum:.4f} | lr {lr:.2e} | grad {grad:.2f} | "
                  f"{rate:,.0f} tok/s | {memory:.1f} GB | "
                  f"ETA {fmt_duration((max_steps - step) * elapsed / max(step - first_step, 1))}",
                  flush=True)
            log.write(json.dumps({"step": step, "loss": loss_sum, "lr": lr, "grad_norm": grad, "tok_per_s": rate,
                                  "epoch": epoch, "time": time.time()}) + "\n")
            if step % cfg["eval_every"] == 0 or step == max_steps:
                val_loss = evaluate(model, val, val_batches, tok.pad_token_id, device)
                record = {"step": step, "val_loss": val_loss, "time": time.time()}
                if retention_batches:
                    record["retention_loss"] = evaluate(model, retention, retention_batches, tok.pad_token_id, device)
                print(f"[qlora] validation loss {val_loss:.4f} (best {min(best, val_loss):.4f})"
                      + (f"; retention loss {record['retention_loss']:.4f}" if retention_batches else ""), flush=True)
                log.write(json.dumps(record) + "\n")
                if val_loss < best:
                    best = val_loss
                    save("adapter_best")
            log.flush()
            if time.time() - last_ckpt > cfg["ckpt_every_minutes"] * 60:
                save()
                last_ckpt = time.time()
            if args.max_hours and time.time() - started > args.max_hours * 3600 and step < max_steps:
                print(f"[qlora] --max-hours {args.max_hours} reached at step {step}: saving; run the same command "
                      "to resume", flush=True)
                break
    except KeyboardInterrupt:
        print("[qlora] interrupted: saving; run the same command to resume", flush=True)
    finally:
        log.close()
        # A run that trained no step (e.g. a finished run started again, as the GPU queue does when the run
        # folder was trained elsewhere) changes nothing, so it must not overwrite that run's record.
        if step > first_step:
            save()
            meta.update({"steps_done": step, "max_steps": max_steps, "best_val_loss": best,
                         "tok_per_s": tokens_seen / max(time.time() - started, 1e-9),
                         "peak_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None,
                         "hardware": hardware, "fp16_loss_scaling": fp16})
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[qlora] {'done' if step >= max_steps else 'stopped'} at step {step} of {max_steps}; "
          f"best validation loss {best:.4f}; adapters in {out}", flush=True)


if __name__ == "__main__":
    main()
