# Generative AI use log (for the A2 AI declaration)

The tutor allowed generative AI on condition that a separate declaration is submitted. This is the
running record it is written from: what Claude (Anthropic's Claude Code) did, and what the team
checked or decided. Earlier entries from the Mac sessions (the `lm/` pipeline, 15 Oct; reading the
sources and writing the delivery plan, 20 Oct) are in `_context/notes/` on the Mac.

| Date | What Claude did | What the team checked or decided |
|---|---|---|
| 22 Oct 2026 | Set up the Windows PC: installed Miniforge, built both conda environments, ran the tests and the LM smoke test. | Cao Tri Tran ran the session and reviewed the results. |
| 22 Oct 2026 | Wrote `explorer/snapshot.py`, `explorer/cli.py` and their tests (DS1 selection, fetch, checks, manifest) to the plan §6.1 spec, and ran the snapshot. | The selection rule and its parameters come from the team's plan; manifest review and a manual transaction check by Cam Dan Bui are still due. |
| 22 Oct 2026 | Wrote `explorer/external.py` and its tests; downloaded DS4 (BitcoinHeist) and recorded its checksums. | Licence (CC BY 4.0) checked by the team on 20 Oct. |
| 22 Oct 2026 | Started DS5 preparation (`lm.prepare_text`), written on the Mac with Claude's help on 15 Oct. | — |
| 22 Oct 2026 | Wrote and tested the processing and modelling code (features port with a parity test against the prototype, split, injection, baseline sweep, Isolation Forest, LOF, K-Means, DBSCAN, regression, evaluation and selection rule, prediction and template explanations), DS2 crawl, DS3/DS4 processing and classification, TxGPT tokenizer, data builder, scoring and baselines; ran DS4 and DS3 classification. | Rules, grids and protocols come from the team's plan; deviations are logged in CHANGELOG.md for the team to accept. |
| 22 Oct 2026 | Diagnosed and fixed operational problems: the pretraining benchmark's out-of-memory bug, Blockstream's rate cap (failover), mempool.space throttling (gzip), a frozen console. | Cao Tri Tran supervised; a second Claude session downloaded Elliptic with the user's Kaggle download. |

| 23 Oct 2026 | Ran the DS1 pipeline end to end (features to evaluation); wrote and ran the label check (`explorer/label_check.py`) and the reviewer sheet and score (`explorer/reviewer.py`), with tests. | The selection rule's recommendation (baseline) and CR-03 (baseline threshold) are for the team to accept; the reviewer ratings will be the team's own. |
| 23 Oct 2026 | Wrote the Assistant pipeline: DS7 builder (`explorer/ds7.py`), the guardrail and measures (`explorer/assistant.py`), the QLoRA trainer (`lm/qlora.py`), the DS6 subset (`lm/chat_subset.py`) and the evaluation (`lm/eval_explain.py`); checked Qwen's licence; downloaded the base weights. | Cao Tri Tran chose Qwen2.5-3B-Instruct (instead of Llama 3.2 3B) and the roles: the 152M model explains DS1 transactions, Qwen is the chatbot and the second writer. The team is to be told. |
| 23 Oct 2026 | Wrote `explorer/plots.py` (17 figures, statistics in `eda_tests.csv`) and `notebooks/01_eda.ipynb`, including the "so what" notes; reviewed every figure as an image and corrected overlaps and one misleading recall chart (leave-one-family-out now shows each model's false-positive rate). | The notes' wording and findings are for the team to review before they go into the report. |
| 23 Oct 2026 | Installed MiKTeX; rebuilt the plan PDF after the Qwen changes; diagnosed the DS2 crawl's speed (measured: the server's bandwidth, not the worker count, is the limit). | Cao Tri Tran decided not to pay for a Blockstream key and to keep the 5,000-address target. |
| 23 Oct 2026 | Proof-checked the day's work against the data, the plan and the rubric: fixed a decoding setting that would have hurt the 152M model's number faithfulness, a contradictory DS7 phrasing, a wrong ETA, and three inaccurate notebook notes; added a memory/speed bench and a chat-retention check to the Qwen trainer; measured the GPU's thermal throttling. | Cao Tri Tran asked for the check; cooling and torch.compile are left to Cao Tri Tran's decision. |
