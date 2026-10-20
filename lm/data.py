"""Batches of training windows read straight from token files on disk.

Token files are flat arrays of uint16 ids (vocabularies up to 65,536), read with
numpy.memmap so a multi-gigabyte corpus never has to fit in RAM.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .model import IGNORE_INDEX

TOKEN_DTYPE = np.uint16


def write_tokens(fh, ids) -> int:
    """Append token ids to an open binary file; return how many were written."""
    arr = np.asarray(ids, dtype=TOKEN_DTYPE)
    fh.write(arr.tobytes())
    return int(arr.size)


def count_tokens(path: Path) -> int:
    return os.path.getsize(path) // np.dtype(TOKEN_DTYPE).itemsize


class WindowSampler:
    """Hands out training sequences in a fixed, shuffled order that can be resumed.

    The token file is cut into consecutive windows of seq_len tokens; window w
    gives inputs tokens[w*T : w*T+T] and targets shifted by one. The windows
    form one long shuffled stream: each epoch visits every window exactly once,
    in an order fixed by (seed, epoch). batch(start, count) returns stream
    positions start .. start+count-1. Training asks for positions by step
    number, so a restarted run sees exactly the sequences the interrupted run
    would have, even with a different micro-batch size, and there is no data
    loader state to save.

    If a mask file is given (one uint8 per token, 1 = train on this token),
    targets whose mask is 0 become IGNORE_INDEX. The chat data uses this so the
    model only learns to write the assistant's replies.
    """

    def __init__(self, tokens_path, seq_len, seed=0, mask_path=None):
        self.tokens_path = str(tokens_path)
        self.mask_path = str(mask_path) if mask_path else None
        self.seq_len = seq_len
        self.seed = seed
        self.n_windows = (count_tokens(Path(tokens_path)) - 1) // seq_len
        if self.n_windows < 1:
            raise ValueError(f"{tokens_path} holds fewer than {seq_len + 1} tokens")
        self._perm_epoch = None
        self._perm = None

    def _order(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def windows(self, start: int, count: int) -> np.ndarray:
        """Window numbers at stream positions start .. start+count-1."""
        out = np.empty(count, dtype=np.int64)
        for j, pos in enumerate(range(start, start + count)):
            epoch, k = divmod(pos, self.n_windows)
            out[j] = self._order(epoch)[k]
        return out

    def batch(self, start: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.seq_len
        # Re-open the memmap for each batch; holding one open for days slowly grows memory use.
        data = np.memmap(self.tokens_path, dtype=TOKEN_DTYPE, mode="r")
        starts = self.windows(start, count) * T
        x = np.stack([data[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        if self.mask_path:
            mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
            keep = np.stack([mask[s + 1:s + T + 1] for s in starts])
            y[keep == 0] = IGNORE_INDEX
        return torch.from_numpy(x), torch.from_numpy(y)
