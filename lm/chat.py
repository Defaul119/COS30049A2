"""Chat with the fine-tuned text model in the terminal.

Usage:
    python -m lm.chat --model runs/text-150m-chat/model.pt

Commands while chatting: /reset clears the conversation, /quit exits.

reply() is separate from the terminal loop so the Assignment 3 back end can
call it directly.
"""
from __future__ import annotations

import argparse
import sys

import torch

from .generate import generate, load_model, stream_text
from .tokenizer import END, EOT, render_chat, token_id

DEFAULT_SAMPLING = {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.15}


def fit_history(tok, messages, budget):
    """Drop the oldest turns (keeping any system message) until the prompt fits in budget tokens."""
    system = [m for m in messages[:1] if m["role"] == "system"]
    rest = messages[len(system):]
    while rest:
        ids, _ = render_chat(tok, system + rest, add_generation_prompt=True)
        if len(ids) <= budget:
            return ids
        rest = rest[2:] if len(rest) > 2 else rest[1:]
    return render_chat(tok, system, add_generation_prompt=True)[0]


def reply(model, tok, messages, max_new_tokens=300, out=None, generator=None, **sampling) -> str:
    """Generate the assistant's next message for a list of {"role", "content"} messages."""
    settings = {**DEFAULT_SAMPLING, **sampling}
    prompt = fit_history(tok, messages, model.cfg.seq_len - max_new_tokens)
    stops = {token_id(tok, END), token_id(tok, EOT)}
    ids_iter = generate(model, prompt, max_new_tokens, stop_ids=stops, vocab_limit=tok.get_vocab_size(),
                        generator=generator, **settings)
    ids = stream_text(tok, ids_iter, out) if out is not None else list(ids_iter)
    ids = [i for i in ids if i not in stops]
    return tok.decode(ids).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--system", help="optional system message")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING["temperature"])
    p.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING["top_k"])
    p.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_SAMPLING["repetition_penalty"])
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    model, tok = load_model(args.model, args.device)
    gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    sampling = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty}
    start = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(start)
    print(f"Model: {args.model} ({model.num_params() / 1e6:.0f}M parameters). /reset clears, /quit exits.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = list(start)
            print("(conversation cleared)\n")
            continue
        messages.append({"role": "user", "content": user})
        sys.stdout.write("bot> ")
        text = reply(model, tok, messages, args.max_new_tokens, out=sys.stdout, generator=gen, **sampling)
        print("\n")
        messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
