"""The Explorer Assistant's writers: each turns chat messages into a reply, for explorer.assistant.explain.

Serves plan §10.3 (the guarded Assistant) and §12.1 (`explain(tx_result, use_assistant=True)`): one place
builds the writers, so the DS7 evaluation (lm.eval_explain) and the Explorer (explorer.predict) run exactly
the same decoding:

    text150m    the 152M model trained here; runs on CPU or GPU (the markers' CPU environment can run it)
    qwen3b      Qwen2.5-3B-Instruct with our QLoRA adapter; needs CUDA and transformers, peft and
                bitsandbytes (environment-gpu.yml only)

`available()` says whether a writer can run on this machine, so the Explorer drops the ones it cannot
load instead of failing; the template is always the last resort (explorer.assistant.explain).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

Generate = Callable[[list[dict[str, str]]], str]
QWEN_PACKAGES = ("transformers", "peft", "bitsandbytes")


def text150m_generator(checkpoint: str | Path, max_new_tokens: int) -> Generate:
    from .chat import reply
    from .generate import load_model

    model, tok = load_model(checkpoint)  # the tokenizer is the tokenizer.json saved next to the weights
    # Greedy, and no repetition penalty: lm.chat's default penalty also counts the prompt's tokens, so it
    # would push the model away from copying the prompt's numbers, which a faithful explanation must do.
    return lambda messages: reply(model, tok, messages, max_new_tokens=max_new_tokens, temperature=0,
                                  repetition_penalty=1.0)


def qwen_generator(config: str | Path, adapter: str | Path, max_new_tokens: int) -> Generate:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from .qlora import load_config, load_tokenizer

    cfg = load_config(str(config))
    tok = load_tokenizer(cfg)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], revision=cfg["revision"], quantization_config=quant,
                                                torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, str(adapter)).eval()

    @torch.no_grad()
    def generate(messages: list[dict[str, str]]) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    return generate


def available(name: str, files: list[str], root: Path) -> str | None:
    """None if writer `name` can run here, else the reason it cannot (missing files, packages or GPU).

    `files` are relative to `root`, and the reason names them that way.
    """
    missing = [f for f in files if not (root / f).exists()]
    if missing:
        return f"missing {', '.join(missing)}"
    if name == "qwen3b":
        absent = [p for p in QWEN_PACKAGES if importlib.util.find_spec(p) is None]
        if absent:
            return f"needs {', '.join(absent)} (environment-gpu.yml)"
        import torch

        if not torch.cuda.is_available():
            return "needs a CUDA GPU (4-bit loading)"
    return None
