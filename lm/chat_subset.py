"""The DS6 chat subset mixed into the Assistant 3B's QLoRA fine-tune (plan §9, "The Explorer Assistant 3B").

The 152M model's chat stage tokenises smol-smoltalk with our own tokenizer (lm.prepare_chat). The
3B model has its own tokenizer and chat template, so this writes the same conversations as plain
messages (JSON Lines) and lm.qlora renders them. It reads the same smol-smoltalk files from the
same hub mirror, in the same order, and holds out every `val_every`-th conversation the same way,
so the two Assistants' chat data overlap rather than differ. Qwen2.5-3B-Instruct is already
chat-tuned, so this subset is small: it only keeps the model's general chat from degrading while
it learns DS7 ("forgetting"), not teach it to chat.

A conversation without a system message gets a neutral one, so the model never falls back to the
base template's built-in default.

Writes data/qwen/chat_train.jsonl and chat_val.jsonl (every `val_every`-th of the first
`max_conversations`), and chat_retention.jsonl: the next `retention` conversations, never trained on,
whose loss lm.qlora logs to show that fine-tuning on DS7 did not degrade general chat.

Round 2 trains on more conversations but must keep round 1's retention set, or the two rounds'
retention losses would measure different things: --retention-start 2500 keeps conversations
2,500-2,699 held out and takes the training conversations around them.

Usage:
    python -m lm.chat_subset                         # 2,500 conversations into data/qwen/
    python -m lm.chat_subset --max-conversations 2000 --out data/qwen_small
    python -m lm.chat_subset --max-conversations 10000 --retention-start 2500 --out data/qwen_v2   # round 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_chat import hub_conversations

DEFAULT_SYSTEM = "You are a helpful assistant."


def with_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The conversation with a system message first (added only if it has none)."""
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    return clean if clean and clean[0]["role"] == "system" else [{"role": "system", "content": DEFAULT_SYSTEM}] + clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/qwen")
    p.add_argument("--raw", default="data/raw/smol-smoltalk")
    p.add_argument("--max-conversations", type=int, default=2500,
                   help="2,500 x 2 epochs is about 3M tokens: enough to keep general chat, small next to DS7")
    p.add_argument("--val-every", type=int, default=100, help="every Nth conversation is held out (as lm.prepare_chat)")
    p.add_argument("--retention", type=int, default=200,
                   help="conversations right after the subset, never trained on: the retention check's held-out set")
    p.add_argument("--retention-start", type=int,
                   help="index of the first retention conversation (default: right after the subset)")
    args = p.parse_args()
    start = args.max_conversations if args.retention_start is None else args.retention_start

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "retention": 0, "skipped": 0}
    files = {name: open(out / f"chat_{name}.jsonl", "w", encoding="utf-8", newline="\n")
             for name in ("train", "val", "retention")}
    seen = 0  # conversations looked at outside the retention window, skipped ones included (as round 1 counted)
    try:
        for i, messages in enumerate(hub_conversations(Path(args.raw))):
            held = start <= i < start + args.retention
            if not held:
                if seen >= args.max_conversations:
                    if i >= start + args.retention:
                        break
                    continue
                seen += 1
            if not any(m["role"] == "assistant" for m in messages):
                counts["skipped"] += 1
                continue
            split = "retention" if held else "val" if i % args.val_every == 0 else "train"
            files[split].write(json.dumps({"messages": with_system(list(messages))}, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in files.values():
            f.close()
    (out / "chat_meta.json").write_text(json.dumps({"source": "HuggingFaceTB/smol-smoltalk (Apache 2.0)",
                                                    "max_conversations": args.max_conversations,
                                                    "val_every": args.val_every, "retention_start": start,
                                                    **counts}, indent=2) + "\n", encoding="utf-8")
    print(f"[chat-subset] {counts} -> {out}")


if __name__ == "__main__":
    main()
