"""Build the chat fine-tuning data for the text model.

Reads conversations from HuggingFaceTB/smol-smoltalk (a chat dataset made for
small models), writes each in our chat format with our own tokenizer, and saves:
    train.bin, val.bin              token ids (uint16)
    train_mask.bin, val_mask.bin    1 where the model learns to write (assistant replies), else 0
    tokenizer.json                  a copy, so this folder is self-contained
    meta.json                       the dataset manifest

--extra adds your own conversations (JSON Lines, one {"messages": [...]} per
line), for example questions and answers about the project.

Usage:
    python -m lm.prepare_chat
    python -m lm.prepare_chat --max-conversations 50000
    python -m lm.prepare_chat --source jsonl --jsonl chats.jsonl --tokenizer data/smoke_text/tokenizer.json --out data/smoke_chat
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .data import write_tokens
from .tokenizer import load_tokenizer, render_chat


def fit_conversation(tok, messages: list[dict], seq_len: int):
    """Return (ids, mask, truncated) for the longest leading part of a conversation
    that fits in seq_len tokens and ends with an assistant reply, or None if even
    the first reply doesn't fit. About half of smol-smoltalk's conversations are
    longer than 1,024 tokens, so cutting them keeps far more data than dropping them.
    """
    ids, mask, ends = render_chat(tok, messages, return_boundaries=True)
    if len(ids) <= seq_len:
        return ids, mask, False
    for k in range(len(messages) - 1, -1, -1):
        if messages[k]["role"] == "assistant" and ends[k] <= seq_len:
            return ids[:ends[k]], mask[:ends[k]], True
    return None


def hub_conversations(raw_dir: Path) -> Iterator[list]:
    from . import hub

    for name in hub.list_files("smol-smoltalk"):
        print(f"[chat] downloading {name} ...", flush=True)
        path = hub.download("smol-smoltalk", name, raw_dir)
        for _, messages in hub.iter_column(path, "messages"):
            yield messages


def jsonl_conversations(path: Path) -> Iterator[list]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["messages"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default="data/text/tokenizer.json")
    p.add_argument("--out", default="data/chat")
    p.add_argument("--raw", default="data/raw/smol-smoltalk")
    p.add_argument("--source", choices=["hub", "jsonl"], default="hub")
    p.add_argument("--jsonl", help="conversations file when --source jsonl")
    p.add_argument("--extra", nargs="*", default=[], help="extra JSON Lines conversation files to include")
    p.add_argument("--seq-len", type=int, default=1024,
                   help="the model's context length; longer conversations are cut after the last reply that fits")
    p.add_argument("--val-every", type=int, default=100, help="every Nth conversation is held out for validation")
    p.add_argument("--max-conversations", type=int, help="stop after this many (from the main source)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(args.tokenizer)
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")

    if args.source == "hub":
        main_source = hub_conversations(Path(args.raw))
    else:
        if not args.jsonl:
            p.error("--source jsonl needs --jsonl")
        main_source = jsonl_conversations(Path(args.jsonl))

    def all_conversations():
        for i, conv in enumerate(main_source):
            if args.max_conversations is not None and i >= args.max_conversations:
                break
            yield conv
        for extra in args.extra:
            yield from jsonl_conversations(Path(extra))

    stats = {"train": 0, "val": 0, "truncated": 0, "too_long": 0, "invalid": 0}
    tokens = {"train": 0, "val": 0}
    targets = {"train": 0, "val": 0}
    files = {
        split: (open(out / f"{split}.bin", "wb"), open(out / f"{split}_mask.bin", "wb"))
        for split in ("train", "val")
    }
    try:
        for i, messages in enumerate(all_conversations()):
            try:
                if not any(m["role"] == "assistant" for m in messages):
                    raise ValueError("no assistant reply")
                fitted = fit_conversation(tok, messages, args.seq_len)
            except (KeyError, TypeError, ValueError):
                stats["invalid"] += 1
                continue
            if fitted is None:
                stats["too_long"] += 1
                continue
            ids, mask, truncated = fitted
            stats["truncated"] += truncated
            split = "val" if i % args.val_every == 0 else "train"
            fids, fmask = files[split]
            tokens[split] += write_tokens(fids, ids)
            fmask.write(np.asarray(mask, dtype=np.uint8).tobytes())
            targets[split] += sum(mask)
            stats[split] += 1
            if (stats["train"] + stats["val"]) % 50_000 == 0:
                print(f"[chat] {stats['train'] + stats['val']:,} conversations, {tokens['train']:,} training tokens", flush=True)
    finally:
        for fids, fmask in files.values():
            fids.close()
            fmask.close()

    source_info = {"name": "local JSON Lines", "license": "n/a"}
    if args.source == "hub":
        from . import hub
        source_info = {"name": "smol-smoltalk", **hub.SOURCES["smol-smoltalk"]}
    meta = {
        "kind": "chat-sft",
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
        "tokenizer": "tokenizer.json",
        "train_tokens": tokens["train"],
        "val_tokens": tokens["val"],
        "train_target_tokens": targets["train"],
        "conversations": {"train": stats["train"], "val": stats["val"]},
        "truncated_to_fit": stats["truncated"],
        "dropped": {"first_reply_longer_than_seq_len": stats["too_long"], "invalid": stats["invalid"]},
        "seq_len": args.seq_len,
        "source": source_info,
        "extra_files": args.extra,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[chat] done: {stats['train']:,} training and {stats['val']:,} validation conversations "
          f"({tokens['train']:,} training tokens); {stats['truncated']:,} cut to fit, "
          f"{stats['too_long']:,} dropped as too long, {stats['invalid']:,} invalid")


if __name__ == "__main__":
    main()
