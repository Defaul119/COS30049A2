# Language models, round 2: plan (Kaggle only)

Written 26 Oct 2026 at the user's request: train the 152M model on **4-5B more tokens** and Qwen 2.5 3B for
**another 12-20 GPU hours**, with every data-building, training and evaluation step on **Kaggle**, not on the
home PC or the Mac, because the user is not happy with the language models' accuracy. Only code edits, `pytest`
and git stay local. This reverses the 24 Oct direction ("no more GPU time for the 152M model"; context.txt).
The results-v1 models stay the fallback until round 2 beats them by the rule in section 3.

## 1. What "accuracy" means today (measured 26 Oct from the saved answers)

Source: `reports/tables/assistant_eval_{text150m,qwen3b}.csv` joined with `data/processed/ds7/test.jsonl`.

| DS7 test (450 rows) | 152M | Qwen 3B | Reading |
|---|---|---|---|
| Explanations: number faithfulness | 99.8% | 100% | one 152M answer invents "120 ... against a usual 85.0" (test-00257) |
| Explanations: guardrail pass | 99.8% | 100% | |
| Wrong comparison word (ad hoc, 25 Oct) | 1 of 445 | 4 of 445 | "Quieter week" when busier; not yet a metric |
| Flagged answers that copy the template | 0% | 96% | Qwen echoes phrasing 0; the 152M rephrases |
| ROUGE-L vs template / vs the row's own reply (445) | 0.683 / 0.719 | 0.783 / 0.744 | DS7 teaches 5 phrasings, so ROUGE-L vs the template penalises correct paraphrases (152M "unscored" answers score 0.23 and are right) |
| **Open questions (glossary): correct** | **0 of 5** | **0 of 5** | Qwen says the Explorer uses Isolation Forest; the 152M invents "Robusta" |

So the explanations are already near the ceiling of the current measures, and the real weakness is
**open questions**, which the test set barely measures (5 prompts on 2 topics). Two consequences for the plan:

- Round 2 must measure what it is meant to improve **before** training starts: a held-out glossary test
  (section 4), a direction check and a copy rate (section 5). Without them the extra GPU hours cannot show a gain.
- More pretraining tokens mostly lower the 152M model's perplexity; facts about this project come from the
  glossary data. Qwen already overfits its current data (best validation loss 0.0752, final 0.0920), so its extra
  hours must go into **more data**, not more epochs of the same data.

## 2. Kaggle budget (the binding constraint)

Kaggle facts: GPU sessions end at **12 h** (we stop at 11 h, as the Qwen run did); only `/kaggle/working`
(**20 GB**) is saved; the weekly GPU quota is not published (users report about **30 h**, reset Saturday
00:00 UTC = 07:00 GMT+7) and is shown in each account's Settings. Check there, and check whether "GPU T4 x2"
counts as one hour per hour (assumed below).

**Speed.** The RTX 3060 trained the 152M model at 11,967 tokens/s (bf16). A T4 has no bf16 (fp16 with loss
scaling, already in `lm/train.py`); our only T4 measurement is Qwen's 326 tokens/s. Estimates, to be replaced
by the session-1 benchmark: one T4 about 7,500 tokens/s; T4 x2 with data parallelism about 13,500 tokens/s.
`lm/train.py` uses one GPU today, so the second T4 needs code (C2).

| 152M extra tokens | one T4 (7.5k tok/s) | T4 x2, DDP (13.5k tok/s) |
|---|---|---|
| 4.0B | 148 h, 15 sessions | 82 h, 8 sessions |
| 4.5B | 167 h, 16 sessions | 93 h, 9 sessions |
| 5.0B | 185 h, 18 sessions | 103 h, 10 sessions |

(Sessions = 11 h, of which about 10.5 h train.) **Total for round 2 with T4 x2 and 4.5B: about 125-135 GPU
hours**: pretraining 93 + session overhead 5 + 152M chat and explanation stages 3 + Qwen 12-20 + Qwen
evaluation 2.5 + benchmarks and reruns about 10.

**Quota available.** The user's account has about 20 h left this week (the 24-25 Oct Qwen run used about 10 h;
check Settings), then 30 h from Sat 31 Oct. One account gives about 50 h before the cut-off, so **round 2 needs
the other three members' Kaggle accounts** (each phone-verified, about 30 h each this week and next).
With two accounts, or without DDP, the gate (section 6) cuts the 152M run to about 2.5B tokens.

**Wall clock.** Pretraining is one chain of sessions (each resumes from the last checkpoint), so accounts take
turns; it cannot be split in parallel. 9 sessions with handoffs take about 4.5 days. Qwen runs on a different
account at the same time.

## 3. Rule to adopt a round-2 model (fixed now, before any result)

On the frozen tests (DS7 v1 test, 445 scored prompts; DS10, 291; the new glossary test), a round-2 writer
replaces its round-1 version in `settings.json -> assistant -> writerOrder` only if **all** hold:

1. guardrail pass rate and number faithfulness are no lower than round 1;
2. wrong comparison words are no more than round 1;
3. open-question key-fact accuracy (section 5) is at least **10 points** higher.

Otherwise round 1 stays and round 2 is reported as an experiment ("3x the pretraining tokens changed X by Y").
Either way the report gets a clean comparison on identical tests. The 152M model's architecture stays the same,
so it still runs in the CPU environment. Its pretraining is also judged on its own: validation loss on the
**same** `val.bin` as round 1 (3.1305). A Chinchilla-style fit predicts about 0.27-0.30 lower at 6-7B total
tokens; that is an expectation, not a target.

## 4. Datasets: scale for round 2

| Data | Round 1 | Round 2 | How |
|---|---|---|---|
| DS5 FineWeb-Edu (152M pretraining) | 2.0B tokens (+10M validation) of the 10B-token sample | **+4-5B new tokens** from the same sample, no document repeated; same tokenizer; same `val.bin` | `prepare_text --continue-from data/text` (C4) in a Kaggle CPU notebook; 8-10 GB of uint16 |
| DS6 smol-smoltalk, 152M chat stage | 100M tokens | unchanged | `prepare_chat` on Kaggle |
| DS6 subset for Qwen (retention) | 2,500 conversations + 200 held out | **10,000** + the same 200 held out | `chat_subset --max-conversations 10000` |
| DS7 explanation pairs | 10,160 (9,149 / 561 / 450) | **about 40,000** in `data/processed/ds7_v2/` (`targetPairs` 40,000, `maxPairsPerAddress` 15); the v1 test file stays the frozen test | `build-ds7 --out` (C7); split by address hash, so v1 test addresses stay test; the build asserts none is in v2 train or val |
| Glossary (open questions) | 160 pairs; 150 / 5 / 5 by topic (2 test topics) | **about 300 pairs, 30+ topics, each with its key facts; 8+ topics (24+ prompts) held out** | `tools/make_glossary.py` (C8); **needs a human read-through**: these are factual claims about the project |

Why these sizes: Qwen trains at about 2 h per epoch of round 1's 11,600 examples, so 12-20 h covers 1.5-2
epochs of about 50,000 examples (40,000 DS7 + 300 glossary, repeated 4x + 10,000 DS6). Fewer passes over
more data is the fix for the overfitting seen in round 1.

## 5. Code before any GPU time (local edits, `pytest`, push; Claude)

| # | Change | Test |
|---|---|---|
| C1 | `lm/train.py --max-hours H`: save `ckpt_last.pt` and exit 0 before Kaggle's 12 h (as `lm.qlora` does) | unit test with a tiny model |
| C2 | `lm/train.py` data parallelism under `torchrun` (rank 0 evaluates and saves; per-rank sampler seed; accumulation split across GPUs). Kept only if the T4 x2 benchmark is at least 1.6x one T4 | 2-process CPU (gloo) smoke test |
| C3 | `lm/train.py` `"schedule": "wsd"` (warm-up, constant, then linear decay from `decay_start_tokens`, re-read at each resume), so the end point can be set late when the quota is known | `lr_at` unit test |
| C4 | `lm/prepare_text.py --continue-from DIR`: copy its tokenizer and `val.bin`, start at its `state.json` next file and row, record the continuation in `meta.json` | `--source local` test |
| C5 | configs `text_150m_pretrain_cont.json` (init from `runs/text-150m/model.pt`, data `data/text_cont`, 4.5B tokens, peak lr 3e-4 after 1,000 warm-up steps, min lr 3e-5, wsd), `text_150m_chat_v2.json`, `text_150m_explain_v2.json`, `qwen3b_explain_v2.json` (fresh from the base model, 2 epochs, early stopping on validation loss) | config load test |
| C6 | direction check (NEXT 1 in context.txt): `explorer.assistant.check` rejects a comparison word that disagrees with the feature's z sign; `lm/eval_explain` adds `direction_ok`, `copy`, `rouge_own` and glossary `key_facts` columns; re-score the saved round-1 answers so round 1 and round 2 use the same measures | unit tests on the known cases (test-00285, test-00127, ds10-00153) |
| C7 | `explorer/ds7.py --out`, settings `ds7v2`; assert no v1-test address in v2 train or val | DS7 test |
| C8 | `tools/make_glossary.py`: 300 pairs with a `key_facts` list per topic (numbers still read from settings) | existing glossary checks |
| C9 | Kaggle notebooks (pinned commit, `MAX_HOURS = 11`, `RESUME_FROM`, `kaggle_run.json` as before): `kaggle/lm_data.ipynb` (CPU: DS5 continuation, DS7 v2, chat data), `kaggle/text150m_train.ipynb` (GPU T4 x2: benchmark, fp16 check, pretraining chain, chat and explanation stages, evaluation), `kaggle/qwen3b_explain.ipynb` v2 | dry run of each notebook's first cells |

The fp16 check: before training, the round-1 checkpoint's validation loss under fp16 on the T4 must equal
the bf16 value (3.1305 +/- 0.01). The model was trained in bf16, and fp16 overflows sooner.

## 6. Steps and dates (GMT+7; due Sun 8 Nov, the plan submits Sat 7 Nov)

| When | Step | Who / where |
|---|---|---|
| Mon 26 Oct | This plan. Ask the three teammates for phone-verified Kaggle accounts and their quota (Settings). Decide C6 and the glossary reviewer | user |
| Tue 27 Oct | C1-C9, `pytest`, push. From the PC, upload a private Kaggle Dataset `cos30049a2-text150m-base`: `runs/text-150m/model.pt`, `data/text/{tokenizer.json, val.bin, meta.json, state.json}` (the one local step left: an upload). Share it with the teammates | Claude; user runs the upload |
| Tue 27 Oct | CPU notebook: DS5 continuation tokens (dataset `cos30049a2-text-cont`), DS7 v2, chat data. CPU sessions do not use GPU quota | Kaggle CPU |
| Wed 28 Oct, 08:00 | GPU session 1: benchmark one T4 vs T4 x2, fp16 check, then **the gate**: `total_tokens` = min(5B, measured tokens/s x GPU hours left before the cut-off), then train | Kaggle T4 x2 |
| Wed 28 Oct - Sun 1 Nov | Pretraining chain: about 9 sessions, handoffs about 07:30 and 19:00. Each output becomes a new version of dataset `cos30049a2-text150m-cont`, attached as `RESUME_FROM` by the next session, on whichever account has quota. Set `decay_start_tokens` by Sat 31 Oct evening at the latest | user + teammates |
| Wed 28 - Fri 30 Oct | Qwen v2 on a second account: benchmark, 2 sessions of up to 11 h (12-20 h in all), then evaluation on DS7 v1 test, DS10 and the glossary test | Kaggle T4 |
| **Sun 1 Nov 23:59** | **Pretraining cut-off**: whatever the chain has reached, decayed, goes on | |
| Mon 2 Nov | One session: 152M chat stage (DS6, 100M tokens), explanation stage (DS7 v2), evaluation on the same three tests | Kaggle T4 x2 |
| Tue 3 Nov | Apply the rule (section 3), copy the tables into `reports/tables/`, models into `models/` (git-ignored; GitHub release assets only with the user's OK), CHANGELOG, tag `results-v1.1` if any number changed | Claude |
| Wed 4 Nov | M4: a teammate rebuilds from the README on a fresh clone; code freeze | teammate |
| Thu 5 - Sat 7 Nov | The report quotes the frozen tables; submit Sat 7 Nov | team |

The report does not wait for round 2: `docs/REPORT_PLAN.md` gives each language model a sentence or a table
row, written now from round 1 and updated on Tue 3 Nov only if section 3 adopts a round-2 model.

## 7. Risks

| Risk | Effect | Response |
|---|---|---|
| Fewer than four Kaggle accounts, or less quota than 30 h | 152M run shorter | the gate sets the tokens; the WSD decay still ends the run cleanly |
| No DDP speed-up on T4 x2 | about half the tokens | one T4 per session; the gate cuts to about 2.5B |
| fp16 overflow with the bf16-trained checkpoint | NaN loss | the fp16 check catches it before training; fall back to a lower peak lr, then fp32 (slower) |
| A session hits 12 h, or a handoff is missed | lost output, or idle hours | `--max-hours 11`; checkpoints every 30 min; two fixed handoff times a day |
| Qwen overfits again | no gain | 4-5x data, 2 epochs at most, best adapter by validation loss |
| Glossary not reviewed in time | no open-question test | without it the rule cannot adopt a round-2 model; the glossary review is the critical human step |
| `/kaggle/working` over 20 GB | output lost | keep only `ckpt_last.pt` (about 1.8 GB), `model_best.pt` and logs per session; read data from `/kaggle/input` |
| Round-2 work crowds out the report (12 marks) and integration | lower marks where they count | Kaggle runs unattended except handoffs; the report goes on in parallel; Tue 3 Nov is the last day round 2 can change anything |

Open questions answer better with **retrieval** (put the matching glossary answers or README lines into the
prompt) than with more training: a cheap alternative or complement if the glossary test still fails after round 2.
