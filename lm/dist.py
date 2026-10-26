"""Data parallelism over the GPUs of one machine (Kaggle's "GPU T4 x2"), for lm.qlora and lm.train.

Serves docs/KAGGLE_LM_PLAN.md (C2; round 2 runs on Kaggle only): a T4 x2 session costs the same quota
as one T4, so a trainer that uses one GPU wastes half of every hour. Started under torchrun, each
process trains on its own share of the micro-batches, and the trainable parameters' gradients are
averaged across processes before every optimizer step, so every process takes the same step and the
weights stay identical. This is what DistributedDataParallel does, done by hand: both trainers keep
their single-GPU loop (gradient accumulation, fp16 loss scaling, clipping, resume), and a QLoRA model
with gradient checkpointing needs none of DDP's hooks.

Without torchrun (WORLD_SIZE unset) everything here is a no-op for one process, so a single-GPU or
CPU run behaves exactly as before.

    torchrun --standalone --nproc_per_node 2 -m lm.qlora --config ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def main(self) -> bool:
        """The process that prints, evaluates the shared result and writes files."""
        return self.rank == 0

    @property
    def on(self) -> bool:
        return self.world > 1


def setup(device_type: str) -> Dist:
    """Join the process group torchrun started (NCCL on GPUs, gloo on the CPU), or a one-process Dist."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return Dist()
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    return Dist(int(os.environ["RANK"]), world, local_rank)


def broadcast_parameters(d: Dist, params: list[torch.Tensor]) -> None:
    """Copy rank 0's values into every process (e.g. LoRA's random initialisation, which differs by process)."""
    if d.on:
        for p in params:
            dist.broadcast(p.data, src=0)


def average_gradients(d: Dist, params: list[torch.Tensor]) -> None:
    """Mean of each parameter's gradient over the processes, in place.

    With fp16 loss scaling this runs before unscaling: an inf or NaN in any process then reaches every
    process, so all of them skip the same step and the weights stay identical.
    """
    if not d.on:
        return
    grads = [p.grad for p in params if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= d.world
    offset = 0
    for g in grads:
        g.copy_(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()


def total(d: Dist, *values: float, device: torch.device) -> list[float]:
    """Each value summed over the processes (token counts, loss sums)."""
    if not d.on:
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def any_true(d: Dist, flag: bool, device: torch.device) -> bool:
    """True in every process if it is true in any: one process's clock decides a time limit for all."""
    return bool(total(d, float(flag), device=device)[0])


def barrier(d: Dist) -> None:
    if d.on:
        dist.barrier()


def cleanup(d: Dist) -> None:
    if d.on:
        dist.destroy_process_group()
