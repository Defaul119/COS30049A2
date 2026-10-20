"""Our own byte-level BPE tokenizer, plus the chat format.

The tokenizer is trained from scratch on the start of the pretraining corpus
(the same algorithm GPT-2 uses), so no part of the text model comes from an
existing model.
"""
from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Special tokens get the first ids (0-4).
EOT = "<|endoftext|>"   # starts every document and every conversation
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"         # closes one message
SPECIAL_TOKENS = [EOT, SYSTEM, USER, ASSISTANT, END]
ROLE_TAGS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        # Start from all 256 byte symbols, so any text, in any language, can be encoded.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_tokenizer(path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def token_id(tok: Tokenizer, token: str) -> int:
    i = tok.token_to_id(token)
    if i is None:
        raise ValueError(f"tokenizer has no {token} token")
    return i


def clean(text: str) -> str:
    """Remove special-token strings from raw text so a web page can't fake a chat turn."""
    for s in SPECIAL_TOKENS:
        text = text.replace(s, "")
    return text


def encode_batch(tok: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Encode many texts at once (the tokenizer library runs this on all CPU cores)."""
    return [e.ids for e in tok.encode_batch([clean(t) for t in texts])]


def render_chat(tok: Tokenizer, messages: list[dict], add_generation_prompt: bool = False,
                return_boundaries: bool = False):
    """Turn a conversation into token ids and a training mask.

    Layout: <|endoftext|> then, for each message, <|role|> content <|end|>.
    The mask is 1 on the assistant's content and its closing <|end|> (what the
    model should learn to write) and 0 elsewhere. With add_generation_prompt,
    a final <|assistant|> tag asks the model to reply. With return_boundaries,
    also return the token position where each message ends.
    """
    ids, mask, ends = [token_id(tok, EOT)], [0], []
    end_id = token_id(tok, END)
    for m in messages:
        role = m["role"]
        if role not in ROLE_TAGS:
            raise ValueError(f"unknown role {role!r}")
        content = tok.encode(clean(m["content"])).ids + [end_id]
        ids.append(token_id(tok, ROLE_TAGS[role]))
        mask.append(0)
        ids.extend(content)
        mask.extend([1 if role == "assistant" else 0] * len(content))
        ends.append(len(ids))
    if add_generation_prompt:
        ids.append(token_id(tok, ASSISTANT))
        mask.append(0)
    return (ids, mask, ends) if return_boundaries else (ids, mask)
