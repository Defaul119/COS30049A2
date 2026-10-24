"""Build the chat fine-tuning data for the text model.

Reads conversations from HuggingFaceTB/smol-smoltalk (a chat dataset made for
small models), writes each in our chat format with our own tokenizer, and saves:
    train.bin, val.bin              token ids (uint16)
    train_mask.bin, val_mask.bin    1 where the model learns to write (assistant replies), else 0
    tokenizer.json                  a copy, so this folder is self-contained
    meta.json                       the dataset manifest

Whole conversations are packed into windows of seq_len tokens (best fit, longest first, over a
buffer of conversations), and each window is filled up with <|endoftext|> tokens that are never
targets. lm.data.WindowSampler cuts the files into exactly these windows, so no conversation is
split across two: every reply is learned with its whole conversation in view. That matters most
for the explanation pairs (DS7), whose replies may only repeat numbers given in the prompt.

--extra adds your own conversations (JSON Lines, one {"messages": [...]} per
line), for example questions and answers about the project. --val-jsonl takes the
validation conversations from their own file (DS7's val.jsonl) instead of holding
out every Nth one.

Usage:
    python -m lm.prepare_chat
    python -m lm.prepare_chat --max-conversations 50000
    python -m lm.prepare_chat --source jsonl --jsonl chats.jsonl --tokenizer data/smoke_text/tokenizer.json --out data/smoke_chat
    python -m lm.prepare_chat --source jsonl --jsonl data/processed/ds7/train.jsonl --val-jsonl data/processed/ds7/val.jsonl --out data/ds7_chat
"""
from __future__ import annotations

import argparse
import json
import shutil
from bisect import bisect_left, insort
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .data import write_tokens
from .tokenizer import EOT, load_tokenizer, render_chat, token_id


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


def pack_windows(conversations: list[tuple[list[int], list[int]]], seq_len: int) -> list[list[tuple]]:
    """Best-fit decreasing: group (ids, mask) conversations, each at most seq_len tokens long,
    into windows of at most seq_len tokens. Returns one list of conversations per window."""
    windows: list[list[tuple]] = []
    free: list[tuple[int, int]] = []  # (tokens still free, window number), kept sorted
    for conv in sorted(conversations, key=lambda c: len(c[0]), reverse=True):
        k = bisect_left(free, (len(conv[0]), -1))  # the fullest window it still fits in
        if k == len(free):
            windows.append([])
            space, w = seq_len, len(windows) - 1
        else:
            space, w = free.pop(k)
        windows[w].append(conv)
        insort(free, (space - len(conv[0]), w))
    return windows


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
    p.add_argument("--val-jsonl", help="validation conversations from this file (then --val-every holds out none)")
    p.add_argument("--seq-len", type=int, default=1024,
                   help="the model's context length; longer conversations are cut after the last reply that fits")
    p.add_argument("--val-every", type=int, default=100, help="every Nth conversation is held out for validation")
    p.add_argument("--max-conversations", type=int, help="stop after this many (from the main source)")
    p.add_argument("--pack-buffer", type=int, default=4096,
                   help="conversations packed together; a larger buffer wastes fewer tokens on filler")
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
        """(split, messages) for every conversation, in source order."""
        for i, conv in enumerate(main_source):
            if args.max_conversations is not None and i >= args.max_conversations:
                break
            yield ("val" if not args.val_jsonl and i % args.val_every == 0 else "train"), conv
        for extra in args.extra:
            for conv in jsonl_conversations(Path(extra)):
                yield "train", conv
        if args.val_jsonl:
            for conv in jsonl_conversations(Path(args.val_jsonl)):
                yield "val", conv

    filler = token_id(tok, EOT)
    stats = {"train": 0, "val": 0, "truncated": 0, "too_long": 0, "invalid": 0}
    tokens = {"train": 0, "val": 0}
    targets = {"train": 0, "val": 0}
    content = {"train": 0, "val": 0}
    windows = {"train": 0, "val": 0}
    buffers: dict[str, list] = {"train": [], "val": []}
    files = {
        split: (open(out / f"{split}.bin", "wb"), open(out / f"{split}_mask.bin", "wb"))
        for split in ("train", "val")
    }

    def flush(split: str) -> None:
        fids, fmask = files[split]
        for window in pack_windows(buffers[split], args.seq_len):
            ids = [t for conv_ids, _ in window for t in conv_ids]
            mask = [m for _, conv_mask in window for m in conv_mask]
            fill = args.seq_len - len(ids)
            content[split] += len(ids)
            tokens[split] += write_tokens(fids, ids + [filler] * fill)
            fmask.write(np.asarray(mask + [0] * fill, dtype=np.uint8).tobytes())
            windows[split] += 1
        buffers[split] = []

    try:
        for split, messages in all_conversations():
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
            buffers[split].append((ids, mask))
            targets[split] += sum(mask)
            stats[split] += 1
            if len(buffers[split]) >= args.pack_buffer:
                flush(split)
            if (stats["train"] + stats["val"]) % 50_000 == 0:
                print(f"[chat] {stats['train'] + stats['val']:,} conversations, {tokens['train']:,} training tokens", flush=True)
        for split, (fids, fmask) in files.items():
            flush(split)
            # One trailing filler token: WindowSampler takes the token after a window as its last
            # target, so without it the final window would never be used.
            tokens[split] += write_tokens(fids, [filler])
            fmask.write(bytes(1))
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
        "windows": windows,
        "filler_share": {k: round(1 - content[k] / max(1, tokens[k]), 4) for k in tokens},
        "truncated_to_fit": stats["truncated"],
        "dropped": {"first_reply_longer_than_seq_len": stats["too_long"], "invalid": stats["invalid"]},
        "seq_len": args.seq_len,
        "source": source_info,
        "extra_files": args.extra,
        "val_file": args.val_jsonl,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[chat] done: {stats['train']:,} training and {stats['val']:,} validation conversations "
          f"({tokens['train']:,} training tokens in {windows['train']:,} windows, "
          f"{meta['filler_share']['train']:.1%} filler); {stats['truncated']:,} cut to fit, "
          f"{stats['too_long']:,} dropped as too long, {stats['invalid']:,} invalid")


if __name__ == "__main__":
    main()
