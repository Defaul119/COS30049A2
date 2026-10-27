"""The Explorer Assistant's writers: each turns chat messages into a reply, for explorer.assistant.explain.

Serves plan §10.3 (the guarded Assistant) and §12.1 (`explain(tx_result, use_assistant=True)`): one place
builds the writers, so the DS7 evaluation (lm.eval_explain) and the Explorer (explorer.predict) run exactly
the same decoding:

    text150m    the 152M model trained from scratch here; runs on CPU or GPU (the markers' CPU environment
                can run it)

`available()` says whether a writer can run on this machine, so the Explorer drops one it cannot load
instead of failing; the template is always the last resort (explorer.assistant.explain).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

Generate = Callable[[list[dict[str, str]]], str]


def text150m_generator(checkpoint: str | Path, max_new_tokens: int) -> Generate:
    from .chat import reply
    from .generate import load_model

    model, tok = load_model(checkpoint)  # the tokenizer is the tokenizer.json saved next to the weights
    # Greedy, and no repetition penalty: lm.chat's default penalty also counts the prompt's tokens, so it
    # would push the model away from copying the prompt's numbers, which a faithful explanation must do.
    return lambda messages: reply(model, tok, messages, max_new_tokens=max_new_tokens, temperature=0,
                                  repetition_penalty=1.0)


def available(files: list[str], root: Path) -> str | None:
    """None if a writer whose `files` (relative to `root`) all exist can run here, else the reason it cannot."""
    missing = [f for f in files if not (root / f).exists()]
    return f"missing {', '.join(missing)}" if missing else None
