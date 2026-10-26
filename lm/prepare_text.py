"""Build the pretraining corpus for the text model.

Steps:
1. Download FineWeb-Edu (its 10B-token sample) from the Hugging Face Hub, one
   Parquet file at a time.
2. Train our own BPE tokenizer on the first --tokenizer-docs documents.
3. Tokenize documents in order: the first --val-tokens tokens go to val.bin
   (held out to measure progress), the rest to train.bin until it holds
   --tokens tokens.
4. Write meta.json, the dataset manifest (source, licence, counts, date).

It is safe to stop with Ctrl+C and run the same command again. Progress is
saved to state.json after every batch of documents, and a re-run continues
from there.

Round 2 (docs/KAGGLE_LM_PLAN.md, C4) continues a finished corpus: --continue-from DIR copies DIR's
tokenizer and validation file and starts at the document after DIR's last one, so the new train.bin
holds only documents the first corpus never used, and validation losses of both rounds are measured
on the same tokens. meta.json records where the continuation started.

Usage:
    python -m lm.prepare_text                        # 2B training tokens, 32,768-token vocabulary
    python -m lm.prepare_text --tokens 200000000     # a smaller corpus
    python -m lm.prepare_text --source local --local-dir path/to/txt --out data/smoke_text   # testing
    python -m lm.prepare_text --continue-from data/text --out data/text_cont --tokens 2000000000 --delete-raw
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from itertools import chain, islice
from pathlib import Path
from typing import Iterator, NamedTuple

from .data import write_tokens
from .tokenizer import EOT, encode_batch, load_tokenizer, token_id, train_tokenizer

BATCH_DOCS = 1024  # documents tokenized together; progress is saved after each batch


class Doc(NamedTuple):
    file: int   # index into the file list
    row: int    # row (document) within that file
    text: str


def hub_docs(files: list[str], raw_dir: Path, start: tuple[int, int], delete_raw: bool) -> Iterator[Doc]:
    from . import hub  # imported here so --source local works without network packages

    start_file, start_row = start
    for fi in range(start_file, len(files)):
        print(f"[data] downloading {files[fi]} ({fi + 1}/{len(files)}) ...", flush=True)
        path = hub.download("fineweb-edu", files[fi], raw_dir)
        first = start_row if fi == start_file else 0
        for row, text in hub.iter_column(path, "text", start_row=first):
            yield Doc(fi, row, text)
        if delete_raw:
            path.unlink(missing_ok=True)


def local_docs(files: list[str], start: tuple[int, int]) -> Iterator[Doc]:
    """Plain-text files for testing: each blank-line-separated block is one document."""
    start_file, start_row = start
    for fi in range(start_file, len(files)):
        blocks = [b.strip() for b in Path(files[fi]).read_text(encoding="utf-8", errors="replace").split("\n\n")]
        blocks = [b for b in blocks if b]
        first = start_row if fi == start_file else 0
        for row in range(first, len(blocks)):
            yield Doc(fi, row, blocks[row])


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/text", help="output folder")
    p.add_argument("--tokens", type=int, default=2_000_000_000, help="training tokens to produce")
    p.add_argument("--val-tokens", type=int, default=10_000_000, help="held-out tokens for validation")
    p.add_argument("--vocab", type=int, default=32768, help="tokenizer vocabulary size (at most 65,536)")
    p.add_argument("--tokenizer-docs", type=int, default=100_000, help="documents used to train the tokenizer")
    p.add_argument("--source", choices=["hub", "local"], default="hub")
    p.add_argument("--local-dir", help="folder of .txt files when --source local")
    p.add_argument("--raw", default="data/raw/fineweb-edu", help="where downloaded Parquet files go")
    p.add_argument("--delete-raw", action="store_true", help="delete each Parquet file once it is tokenized")
    p.add_argument("--continue-from", metavar="DIR",
                   help="a finished corpus to continue: its tokenizer, validation file and file list, from its next document")
    args = p.parse_args()
    if args.vocab > 65536:
        p.error("--vocab must be at most 65,536 so token ids fit in uint16")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok_path, state_path = out / "tokenizer.json", out / "state.json"
    train_path, val_path = out / "train.bin", out / "val.bin"

    state = read_json(state_path)
    if args.continue_from and not state:
        # Start where the finished corpus stopped: same files, same tokenizer, same validation tokens.
        src = Path(args.continue_from)
        previous = read_json(src / "state.json")
        if not (previous and previous.get("finished")):
            raise SystemExit(f"[data] {src} is not a finished corpus (no finished state.json)")
        shutil.copyfile(src / "tokenizer.json", tok_path)
        shutil.copyfile(src / "val.bin", val_path)
        state = {"files": previous["files"], "next_file": previous["next_file"], "next_row": previous["next_row"],
                 "train_tokens": 0, "val_tokens": previous["val_tokens"], "documents": 0,
                 "continued_from": {"dir": src.as_posix(), "next_file": previous["next_file"],
                                    "next_row": previous["next_row"], "train_tokens": previous["train_tokens"],
                                    "documents": previous["documents"]}}
        train_path.unlink(missing_ok=True)
        write_json(state_path, state)
        print(f"[data] continuing {src} from file {state['next_file']}, row {state['next_row']:,}")
    if state and state.get("finished"):
        print(f"[data] already finished: {state['train_tokens']:,} training tokens in {out}. Delete {state_path} to rebuild.")
        return

    # The file list is fixed on the first run and reused, so a resumed run reads files in the same order.
    if state:
        files = state["files"]
    elif args.source == "hub":
        from . import hub
        files = hub.list_files("fineweb-edu")
    else:
        if not args.local_dir:
            p.error("--source local needs --local-dir")
        files = sorted(str(f) for f in Path(args.local_dir).rglob("*.txt"))
    if not files:
        raise SystemExit("[data] no input files found")

    def docs_from(start):
        if args.source == "hub":
            return hub_docs(files, Path(args.raw), start, args.delete_raw)
        return local_docs(files, start)

    if state and tok_path.exists():
        # Resume: cut off anything written after the last saved batch, then continue.
        tok = load_tokenizer(tok_path)
        counts = {"train": state["train_tokens"], "val": state["val_tokens"]}
        for path, n in ((train_path, counts["train"]), (val_path, counts["val"])):
            if not path.exists():
                if n:
                    raise SystemExit(f"[data] {path} is missing; delete {state_path} to start again")
                path.touch()
            with open(path, "r+b") as fh:
                fh.truncate(n * 2)
        n_docs = state["documents"]
        stream = docs_from((state["next_file"], state["next_row"]))
        print(f"[data] resuming at {counts['train']:,} training tokens ({n_docs:,} documents done)")
    else:
        for path in (train_path, val_path):
            path.unlink(missing_ok=True)
        counts, n_docs = {"train": 0, "val": 0}, 0
        stream = docs_from((0, 0))
        if tok_path.exists():
            tok = load_tokenizer(tok_path)
            print(f"[data] using existing tokenizer {tok_path}")
        else:
            head = list(islice(stream, args.tokenizer_docs))
            print(f"[data] training a {args.vocab:,}-token BPE tokenizer on {len(head):,} documents ...", flush=True)
            t0 = time.time()
            tok = train_tokenizer((d.text for d in head), args.vocab)
            tok.save(str(tok_path))
            print(f"[data] tokenizer saved ({tok.get_vocab_size():,} tokens, {time.time() - t0:.0f}s)")
            stream = chain(head, stream)

    eot = token_id(tok, EOT)
    start_time, start_tokens = time.time(), counts["train"]
    continued = (state or {}).get("continued_from")
    if continued:
        args.val_tokens = counts["val"]  # the validation file stays the first corpus's, token for token

    def flush(batch: list[Doc]) -> None:
        nonlocal n_docs
        with open(train_path, "ab") as ftrain, open(val_path, "ab") as fval:
            for ids in encode_batch(tok, [d.text for d in batch]):
                ids = [eot] + ids  # <|endoftext|> marks the start of each document
                if counts["val"] < args.val_tokens:
                    counts["val"] += write_tokens(fval, ids)
                else:
                    counts["train"] += write_tokens(ftrain, ids)
        n_docs += len(batch)
        last = batch[-1]
        write_json(state_path, {
            "files": files, "next_file": last.file, "next_row": last.row + 1,
            "train_tokens": counts["train"], "val_tokens": counts["val"], "documents": n_docs,
            **({"continued_from": continued} if continued else {}),
        })

    batch: list[Doc] = []
    n_batches = 0
    for doc in stream:
        batch.append(doc)
        if len(batch) < BATCH_DOCS:
            continue
        flush(batch)
        batch = []
        n_batches += 1
        if n_batches % 20 == 0:
            done = counts["train"] - start_tokens
            rate = done / max(time.time() - start_time, 1e-9)
            left = (args.tokens - counts["train"]) / max(rate, 1e-9)
            print(f"[data] {counts['train']:,}/{args.tokens:,} training tokens "
                  f"({100 * counts['train'] / args.tokens:.1f}%) | {rate:,.0f} tok/s | ~{left / 60:.0f} min left", flush=True)
        if counts["train"] >= args.tokens:
            break
    if batch:
        flush(batch)

    if counts["train"] < args.tokens:
        print(f"[data] note: the source ran out at {counts['train']:,} training tokens")
    source_info = {"name": "local text files", "license": "n/a"}
    if args.source == "hub":
        from . import hub
        source_info = {"name": "fineweb-edu", **hub.SOURCES["fineweb-edu"]}
    write_json(out / "meta.json", {
        "kind": "text-pretraining",
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot,
        "dtype": "uint16",
        "tokenizer": "tokenizer.json",
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "documents": n_docs,
        "source": source_info,
        "files_used": files[(continued or {}).get("next_file", 0): read_json(state_path)["next_file"] + 1],
        **({"continued_from": continued} if continued else {}),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    state = read_json(state_path)
    state["finished"] = True
    write_json(state_path, state)
    print(f"[data] done: {counts['train']:,} training + {counts['val']:,} validation tokens in {out}")


if __name__ == "__main__":
    main()
