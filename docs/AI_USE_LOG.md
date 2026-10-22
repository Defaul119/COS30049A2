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

