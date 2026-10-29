"""Decoder-only transformer (GPT-style), trained from scratch.

The same model class serves both language models in this project:
- the text chatbot (vocabulary from our own BPE tokenizer), and
- the transaction-sequence model (vocabulary of transaction event tokens).

Design choices, all standard in current small language models:
- Pre-norm blocks with RMSNorm, which train more stably than post-norm.
- Rotary position embeddings (RoPE) on queries and keys, so positions cost no
  parameters.
- A SwiGLU feed-forward layer.
- No bias terms; the input embedding and the output layer share one weight
  matrix ("weight tying"), which saves about 25M parameters at our size.
- torch's scaled_dot_product_attention, which picks the fastest attention
  kernel available on the GPU.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Label value that means "no loss at this position". The chat fine-tuning data
# uses it to skip the user's turns; padding uses it too.
IGNORE_INDEX = -1


@dataclass
class GPTConfig:
    """The model's shape; saved with every checkpoint so it loads without the training config."""
    vocab_size: int = 32768
    seq_len: int = 1024        # longest sequence the model can attend over
    n_layer: int = 18
    n_head: int = 12
    d_model: int = 768
    mlp_hidden: int = 2048     # SwiGLU width, about 8/3 x d_model
    dropout: float = 0.0
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


def pad_vocab(n: int, multiple: int = 64) -> int:
    """Round the vocabulary up so the output layer has GPU-friendly shapes."""
    return ((n + multiple - 1) // multiple) * multiple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in float32 for stability, then return to the input dtype.
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables for rotary embeddings, shape (seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (first half, second half) feature pair of x by its position's angle.

    x: (batch, heads, time, head_dim); cos/sin: (time, head_dim/2).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


class Attention(nn.Module):
    """Causal multi-head self-attention with rotary embeddings on queries and keys."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must divide evenly into heads"
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past=None):
        """past is this layer's (keys, values) from earlier positions, used when generating."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if past is None:
            # Training, or the first pass over a prompt: each position sees itself and earlier ones.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Generating one new token: it may attend to every cached position.
            if T != 1:
                raise ValueError("with a cache, feed exactly one new token at a time")
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """One pre-norm transformer block: RMSNorm, attention, RMSNorm, SwiGLU, each with a residual connection."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, past=None):
        a, present = self.attn(self.norm1(x), cos, sin, past)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, present


class GPT(nn.Module):
    """The decoder-only language model: token embeddings, the blocks, a final RMSNorm and the tied output layer."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Rotary tables are recomputed at start-up, so they are not saved in checkpoints.
        cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # GPT-2 initialisation: N(0, 0.02), with the layers that write into the
        # residual stream scaled down by depth so activations don't grow with n_layer.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past=None, use_cache=False, reduction="mean"):
        """Run the model.

        idx:       (batch, time) token ids.
        targets:   (batch, time) next-token ids, or IGNORE_INDEX to skip a position.
                   If given, logits cover every position and a loss is returned.
                   If not, only the last position's logits are computed (for generation).
        past:      per-layer cached (keys, values) from earlier positions.
        use_cache: also return the updated cache.
        reduction: "mean" for a training loss, "none" for a (batch, time) loss per token,
                   which the transaction model uses as its "how surprising" score.

        Returns (logits, loss, cache); loss and cache are None when not requested.
        """
        B, T = idx.shape
        start = 0 if past is None else past[0][0].shape[2]
        if start + T > self.cfg.seq_len:
            raise ValueError(f"sequence of {start + T} tokens exceeds seq_len {self.cfg.seq_len}")
        cos = self.rope_cos[start:start + T]
        sin = self.rope_sin[start:start + T]

        x = self.drop(self.tok_emb(idx))
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            x, present = block(x, cos, sin, None if past is None else past[i])
            if use_cache:
                presents.append(present)
        x = self.norm(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None, presents

        logits = self.lm_head(x)
        flat_logits = logits.float().view(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        if reduction == "none":
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="none")
            return logits, loss.view(B, T), presents
        # Average over the positions that count. Dividing by at least 1 keeps the loss
        # finite (zero) for a batch in which every position is ignored.
        total = F.cross_entropy(flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="sum")
        count = (flat_targets != IGNORE_INDEX).sum().clamp(min=1)
        return logits, total / count, presents

    def num_params(self) -> int:
        # parameters() lists the shared embedding/output matrix once.
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self) -> float:
        """Training FLOPs per token: 6 per parameter plus the attention score matmuls."""
        c = self.cfg
        return 6 * self.num_params() + 12 * c.n_layer * c.seq_len * c.d_model

    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        """AdamW with weight decay on the matrices only, not on norm gains."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), fused=(device_type == "cuda"))
