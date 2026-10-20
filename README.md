# COS30049 Assignment 2: language models trained from scratch

Group HCMC_COS30049.1_4. This folder holds the two language models, both trained from random weights:

1. **Text chatbot.** A 152M-parameter GPT-style transformer with its own 32,768-token BPE tokenizer.
   It is pretrained on 2 billion tokens of educational web text (FineWeb-Edu), then fine-tuned on
   conversations (smol-smoltalk) so it answers in a chat.
2. **Transaction-sequence model** (next step, not built yet). The same transformer code, trained on
   Bitcoin address histories, scoring how surprising each transaction is.

Nothing here starts from an existing model: the tokenizer, the weights and every training step are ours.

## What it needs

Estimates for the training PC (i5-12600K, 16 GB RAM, RTX 3060 12 GB). Step 4's speed test gives the real
training time.

| Stage | Time | Download | Disk |
|---|---|---|---|
| Prepare pretraining data (2B tokens) | about 30–90 min, plus the download | about 6.5 GB | 4 GB of tokens; the 6.5 GB download can be deleted as it goes |
| Pretrain | about 2.5–4 days nonstop | – | about 3 GB of checkpoints |
| Prepare chat data | about 10–20 min, plus the download | about 0.9 GB | about 2 GB |
| Chat fine-tune (100M tokens) | about 3–4 hours | – | about 3 GB |

Keep about 20 GB free.

## 1. Set up (once)

Install [Miniconda](https://docs.anaconda.com/miniconda/), open **Anaconda Prompt**, and run:

```
cd path\to\asm2
conda env create -f environment-gpu.yml
conda activate cos30049-a2-gpu
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True NVIDIA GeForce RTX 3060`. If it prints `False`, update the NVIDIA driver
and try again.

Run every later command from this folder, with the environment active.

## 2. Check that everything works (about 2 minutes)

```
python -m unittest discover -s tests
python -m lm.smoke
```

The unit tests should end with `OK`. The smoke test runs the whole pipeline on a tiny model with made-up
data and should report `8/8 stages passed`. If anything fails, stop and fix it before the long run.

## 3. Prepare the pretraining data

```
python -m lm.prepare_text --delete-raw
```

This downloads FineWeb-Edu one 2.15 GB file at a time, trains the tokenizer on the first 100,000
documents, and writes `data/text/train.bin` (2B tokens), `val.bin` (10M held-out tokens), `tokenizer.json`
and `meta.json` (the dataset manifest). `--delete-raw` removes each downloaded file once it's tokenized.

If it stops (Ctrl+C, a dropped connection, a restart), run the same command again and it continues.

## 4. Speed test, then pretrain

```
python -m lm.train --config configs/text_150m_pretrain.json --bench
```

This tries several micro-batch sizes and prints the speed, GPU memory and total time for each, then
recommends one. Put that number in `"micro_batch"` in `configs/text_150m_pretrain.json`. Then start
training:

```
python -m lm.train --config configs/text_150m_pretrain.json
```

Each line shows the step, the training loss, the learning rate, tokens per second, how efficiently the GPU
is used (MFU), memory and the time remaining. Every 200 steps it prints the **validation loss** on text the
model never trains on. It starts near 10.4 and should keep falling.

- **To stop:** press Ctrl+C once and wait. It saves a checkpoint, then exits.
- **To continue,** after stopping, a crash or a power cut, run the same command. It resumes from the last
  checkpoint (saved every 30 minutes), with the same data order and learning-rate schedule.
- **Keep Windows awake:** Settings → System → Power → set sleep to *Never* while plugged in. Settings →
  Windows Update → *Pause updates* for a week, so it doesn't restart mid-run.
- **Try the model while it trains:** `python -m lm.generate --model runs/text-150m/model_best.pt --prompt "The water cycle is"`.
  It continues the text. It won't follow instructions until step 5.

When it finishes, the weights are in `runs/text-150m/model.pt`.

## 5. Chat fine-tuning

```
python -m lm.prepare_chat
python -m lm.train --config configs/text_150m_chat.json
```

`prepare_chat` downloads smol-smoltalk (about 460,000 conversations) and writes them in our chat format
with a mask, so the model only learns to write the assistant's replies. About half the conversations are
longer than the model's 1,024-token window, so they're cut after the last reply that fits.
`--extra my_chats.jsonl` adds your own conversations, one `{"messages": [{"role": ..., "content": ...}, ...]}`
per line.

## 6. Chat

```
python -m lm.chat --model runs/text-150m-chat/model.pt
```

Type a message and press Enter. `/reset` clears the conversation and `/quit` exits.

Expect a small model: it writes fluent-looking English, stays roughly on topic, and is often wrong on facts.
A 150M model trained on 2B tokens has seen about a thousandth of the data behind commercial chatbots.

## Troubleshooting

| Problem | Fix |
|---|---|
| `out of memory` | Lower `micro_batch` to 4 or 2. It must divide 256 (`batch_tokens` / 1,024). The results don't change, only the speed. |
| `UserWarning: ... not compiled with flash attention` | Harmless on Windows. PyTorch uses its memory-efficient attention instead. |
| Download errors or `429 Too Many Requests` | Wait a few minutes and run the same command; downloads resume. |
| Want to change settings mid-run | `lr`, `eval_every`, `micro_batch` and the checkpoint interval can change between runs. Changing the model shape or `batch_tokens` needs a new run (`--fresh`). |
| Training says `diverged` | Lower `lr` in the config and run the same command; it resumes from the last good checkpoint. |

## How it works

| File | What it does |
|---|---|
| `lm/model.py` | The transformer: 18 layers, width 768, 12 heads, rotary positions, SwiGLU, RMSNorm, tied embeddings |
| `lm/tokenizer.py` | Trains our byte-level BPE tokenizer; the chat format and its training mask |
| `lm/hub.py` | Downloads datasets from Hugging Face; lists every source and its licence |
| `lm/prepare_text.py` | Builds the pretraining token files (resumable) |
| `lm/prepare_chat.py` | Builds the chat fine-tuning files |
| `lm/data.py` | Reads training windows from disk in a fixed, resumable shuffled order |
| `lm/train.py` | Training: AdamW, warm-up and cosine schedule, bf16, gradient accumulation, checkpoints, speed test |
| `lm/generate.py` | Loading a model and sampling text, with a key/value cache |
| `lm/chat.py` | The chat loop; `reply()` is reusable by the Assignment 3 back end |
| `lm/smoke.py` | The end-to-end check in step 2 |
| `configs/` | Run settings: model size, tokens, batch, learning rate |
| `tests/` | Unit tests |

**Model size:** 152.6M parameters. 18 layers × 7.08M each, plus a 25.2M embedding shared with the output
layer.

**Training recipe:**
- Batches of 262,144 tokens, so 7,630 steps for 2B tokens.
- AdamW (β 0.9/0.95, weight decay 0.1) with gradient clipping at 1.0.
- Learning rate warms up over 500 steps to 6e-4, then follows a cosine decay to 6e-5.
- bf16 mixed precision.

## Data sources and licences

| Dataset | Used for | Licence |
|---|---|---|
| FineWeb-Edu, 10B-token sample (HuggingFaceFW/fineweb-edu) | Pretraining (first ~2B tokens) | ODC-By 1.0 |
| smol-smoltalk (HuggingFaceTB/smol-smoltalk) | Chat fine-tuning | Apache 2.0 |

`data/text/meta.json` and `data/chat/meta.json` record exactly what was used, with counts and dates.

References:
- Penedo, G. et al. (2024) 'The FineWeb datasets: decanting the web for the finest text data at scale',
  *Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track)*.
- Allal, L.B. et al. (2025) 'SmolLM2: when smol goes big — data-centric training of a small language
  model', *arXiv preprint* arXiv:2502.02737.

## Note for the Assignment 2 zip

The brief asks for the processed data used by the final model. The token files (about 6 GB) are too large
for a zip, so the plan is to submit:
- the `meta.json` manifests;
- the training logs (`runs/*/log.jsonl`);
- a small sample of each token file;
- these scripts, which rebuild everything.

Confirm with the tutor that this is acceptable.
