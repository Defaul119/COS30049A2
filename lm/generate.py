"""Load a trained model and generate text with it.

Usage (plain text continuation, useful for checking the pretrained model):
    python -m lm.generate --model runs/text-150m/model.pt --prompt "The water cycle is"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Sequence

import torch

from .model import GPT, GPTConfig
from .tokenizer import EOT, load_tokenizer, token_id


def load_model(path, device="auto"):
    """Return (model, tokenizer) from a model.pt / model_best.pt / ckpt_last.pt file.

    The tokenizer is the tokenizer.json saved next to the weights.
    """
    path = Path(path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    tok_path = path.parent / "tokenizer.json"
    tok = load_tokenizer(tok_path) if tok_path.exists() else None
    return model, tok


def sample_next(logits: torch.Tensor, history: Sequence[int], temperature=0.8, top_k=50, top_p=0.95,
                repetition_penalty=1.0, vocab_limit=None, generator=None) -> int:
    """Pick the next token id from the last position's logits."""
    logits = logits.float().cpu().clone()
    if vocab_limit is not None:
        logits[vocab_limit:] = float("-inf")  # padding ids the tokenizer never produces
    if repetition_penalty != 1.0 and history:
        # Make recently used tokens less likely, which curbs the loops small models fall into.
        recent = torch.tensor(sorted(set(history[-256:])), dtype=torch.long)
        vals = logits[recent]
        logits[recent] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        # Keep the smallest set of tokens whose probabilities add up to top_p.
        sorted_probs, order = torch.sort(probs, descending=True)
        drop = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > top_p
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model: GPT, prompt_ids: Sequence[int], max_new_tokens=256, stop_ids=(), **sampling) -> Iterator[int]:
    """Yield new token ids one at a time, stopping after a stop id or max_new_tokens.

    Keys and values of earlier positions are cached, so each new token costs
    one position's work. When the context window fills up, the most recent
    half is kept and re-read.
    """
    model.eval()
    device = next(model.parameters()).device
    limit = model.cfg.seq_len
    tokens = list(prompt_ids)
    if not tokens:
        raise ValueError("prompt is empty")

    def prefill(seq):
        logits, _, past = model(torch.tensor([seq], dtype=torch.long, device=device), use_cache=True)
        return logits, past

    window = tokens[-limit:]
    logits, past = prefill(window)
    cached = len(window)
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[0, -1], tokens, **sampling)
        yield nxt
        if nxt in stop_ids:
            return
        tokens.append(nxt)
        if cached >= limit:
            window = tokens[-(limit // 2):]
            logits, past = prefill(window)
            cached = len(window)
        else:
            logits, _, past = model(torch.tensor([[nxt]], dtype=torch.long, device=device), past=past, use_cache=True)
            cached += 1


def stream_text(tok, ids_iter, out=sys.stdout) -> list[int]:
    """Print tokens as they are generated; return all generated ids."""
    ids, shown = [], ""
    for i in ids_iter:
        ids.append(i)
        text = tok.decode(ids)
        if not text.endswith("\ufffd"):  # wait until a multi-byte character is complete
            out.write(text[len(shown):])
            out.flush()
            shown = text
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    model, tok = load_model(args.model, args.device)
    gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    eot = token_id(tok, EOT)
    prompt = [eot] + tok.encode(args.prompt).ids
    sys.stdout.write(args.prompt)
    stream_text(tok, generate(model, prompt, args.max_new_tokens, stop_ids={eot}, temperature=args.temperature,
                              top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                              vocab_limit=tok.get_vocab_size(), generator=gen))
    print()


if __name__ == "__main__":
    main()
