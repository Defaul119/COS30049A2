"""Write data/processed/ds7_v2/glossary.jsonl: the glossary pairs of round 2's DS7 (plan §6.7; docs/KAGGLE_LM_PLAN.md C8).

The plan says the team writes about 300 questions and answers about the Explorer. On 23 Oct 2026
the team delegated this to Cao Tri Tran, and Claude drafted them (declared in docs/AI_USE_LOG.md).
On 26 Oct, at Cao Tri Tran's request, Claude checked every answer against the code, settings and
result tables (six corrected; CHANGELOG) in place of a team read-through: STATUS records that.
Each topic has one answer, the key facts a correct answer must state, several taught phrasings
of the question and one or two held-out ones.

Round 1 split the glossary by topic, so its test topics were never taught and a correct answer was
impossible (both models answered 0 of 5). Round 2 asks what the open-question test is for: whether
a model learnt the facts it was taught. So every topic is taught, and its held-out phrasings,
written differently, form the test (split "test", judged by key facts in lm.eval_explain). Round
1's own test questions are held out here too, so the frozen v1 test set stays unseen
(explorer.ds7 stops the build otherwise). Round 1's glossary.jsonl in data/processed/ds7/ is kept
as it was built (git history has its script).

Every number in an answer is read from config/settings.json, models/baseline.json or the DS1
manifest when this script runs, so the glossary cannot drift from the code. The script refuses to
write if an answer uses a banned word, misses the limitation line where one is required, or does
not itself contain every one of its key facts.

Usage:  python tools/make_glossary.py
        (then: python -m explorer.cli build-ds7 --out data/processed/ds7_v2 --settings ds7v2)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
FLAG_Z = json.loads((ROOT / "models" / "baseline.json").read_text(encoding="utf-8"))["chosen"]["refined"]["threshold"]
_DS1_MANIFEST = json.loads((ROOT / "data" / "processed" / "esplora" / "manifest.json").read_text(encoding="utf-8"))
DS1, PICK = _DS1_MANIFEST["totals"], _DS1_MANIFEST["selection_rule"]["parameters"]
F, B, M, INJ = S["features"], S["baseline"], S["model"], S["injection"]
# The threshold sweep (reports/tables/tuning_baseline.csv): addresses over the sanity limit at A1's threshold and ours.
with open(ROOT / "reports" / "tables" / "tuning_baseline.csv", encoding="utf-8") as _f:
    _SWEEP = {float(r["threshold"]): int(r["addresses_over_limit"]) for r in csv.DictReader(_f) if r["features"] == "refined"}
OVER_A1, OVER_OURS = _SWEEP[B["flagAbsZ"]], _SWEEP[FLAG_Z]
LIMIT = S["limitationLine"]
OUT = ROOT / "data" / "processed" / "ds7_v2" / "glossary.jsonl"
# Topics whose answer must end a flag-like statement with the limitation line (A1 NFR-01).
NEEDS_LIMIT = ("not-a-judgement", "dont-know", "reading-explanation", "what-next", "behaviour-change")
# One in VAL_EVERY topics lends its last taught phrasing to validation, so the best adapter is also chosen
# on glossary answers, not only explanations.
VAL_EVERY = 3
STATUS = "fact-checked against code and tables by Claude, 26 Oct 2026, at the user's request"


def n(x: float) -> str:
    """A number as the answers print it: 5.0 -> 5, 0.02 -> 0.02."""
    return f"{x:g}"


class Topic(NamedTuple):
    name: str
    answer: str
    key_facts: list[str | list[str]]   # each a phrase, or interchangeable phrases; see explorer.assistant.key_facts_found
    taught: list[str]
    held_out: list[str]


TOPICS: list[Topic] = [
    Topic("what-is-explorer",
          "The Explorer shows a Bitcoin address's transaction history and ranks each transaction by how unusual it is "
          "compared with that same address's own past. For each flagged transaction it gives a plain-language reason. "
          "It is an investigation aid that points to transactions worth a look; it does not judge anyone.",
          ["unusual", ["own past", "own history"], ["investigation aid", "worth a look"]],
          ["What is the Bitcoin Address Activity Explorer?", "What does this tool do?", "What is the Explorer for?",
           "Can you explain what this app is?", "Give me a short description of the Explorer."],
          ["In one sentence, what does the Explorer do?", "I'm new here. What is this tool supposed to help me with?"]),
    Topic("intended-users",
          "It is for analysts and investigators who need to review an address's activity quickly, for example a "
          "compliance analyst or a researcher. It helps them decide where to look first; the judgement stays with them.",
          [["analyst", "investigator"], ["where to look first", "judgement stays"]],
          ["Who is the Explorer for?", "Who are the intended users?", "Who should use this tool?",
           "Which people is this tool designed for?"],
          ["Is this tool meant for professionals or for anyone?"]),
    Topic("not-a-judgement",
          "No. A flag only means the transaction is unusual for this address's own history: larger, faster, busier or "
          "differently shaped than usual. There are many ordinary reasons for that, such as a one-off purchase or a "
          f"change of wallet software. {LIMIT}",
          [["own history", "own past"], "ordinary reason", LIMIT],
          ["Does a flag mean the owner did something wrong?", "Is a flagged transaction proof of wrongdoing?",
           "Is this address a criminal?", "Does this mean the address is involved in fraud?",
           "Should I treat a flagged address as dangerous?"],
          ["My transaction got flagged. Am I in trouble?", "So the flagged ones are the bad actors, right?"]),
    Topic("who-owns-address",
          "No. The Explorer only uses public blockchain data: amounts, times and the shape of transactions. It never "
          "says who controls an address or why it transacts, and it should not be used to guess.",
          ["public blockchain data", "who controls"],
          ["Who owns this address?", "Can you tell me whose wallet this is?",
           "Can the Explorer identify the person behind an address?", "Can you find the real name behind this wallet?"],
          ["Whose bitcoin is this?", "Can the tool de-anonymise an address for me?"]),
    Topic("why-unusual-not-wrongdoing",
          "Because the evidence says they are different things. When we scored two labelled public datasets with our "
          "unusualness methods, the labelled positive transactions were, if anything, less unusual than the rest. "
          f"Unusual is a reason to look, not a conclusion. {LIMIT}",
          [["labelled", "labels"], "less unusual", "reason to look"],
          ["Why don't you call unusual transactions bad?", "Why is unusual not the same as wrongdoing?",
           "Why does the Explorer avoid accusing anyone?", "What is the difference between unusual and wrong?"],
          ["If something is unusual, isn't it probably bad?"]),
    Topic("limitation-line",
          f"Every flagged item ends with: \"{LIMIT}\" It is there so that nobody reads a flag as a verdict. The Explorer "
          "measures how different a transaction is from the address's past, nothing more.",
          [LIMIT, "verdict"],
          ["What does the line at the end of a flag mean?", "Why does every flag end with the same sentence?",
           "What is the limitation line?", "Why do flags always carry a warning sentence?"],
          ["What's that sentence at the bottom of every flagged result?"]),
    Topic("data-source",
          "From public Bitcoin blockchain data, fetched through the Esplora API (Blockstream, with mempool.space as a "
          "fallback when Blockstream rate-limits). The Explorer works on a saved snapshot, so results can be reproduced.",
          ["Esplora", "mempool.space", "snapshot"],
          ["Where does the data come from?", "What data does the Explorer use?", "Which API is the data from?",
           "Which blockchain API does the Explorer call?"],
          ["How does the Explorer get its transactions?", "Is the data from an exchange or from the blockchain?"]),
    Topic("snapshot",
          f"It uses a fixed snapshot of {DS1['main_addresses']} addresses with long histories and "
          f"{DS1['short_addresses']} with short ones, saved once with checksums. A snapshot makes every result "
          "reproducible and avoids depending on a public API's limits. New transactions after the snapshot are not shown.",
          ["snapshot", "reproducible", ["not shown", "after the snapshot"]],
          ["Why is the data a snapshot?", "Is the data live?", "Does the Explorer show the latest transactions?",
           "Why doesn't the Explorer fetch data live?"],
          ["Will I see a transaction that happened this morning?"]),
    Topic("own-history",
          "With the same address's own earlier transactions. Addresses differ hugely: in our snapshot the typical "
          "amount of one address is thousands of times that of another, so one global threshold would flag the big "
          "addresses all the time and the small ones never.",
          [["same address", "own earlier"], "global threshold"],
          ["Compared with what is a transaction unusual?", "What is the transaction compared against?",
           "Why compare an address only with itself?", "Why not use one threshold for every address?"],
          ["What baseline is each transaction measured against?"]),
    Topic("strictly-earlier",
          "No. A transaction is compared only with transactions that happened strictly before it. Transactions in the "
          "same block are never each other's history. This keeps every score something an analyst could have "
          "computed at that moment.",
          ["strictly before", "same block"],
          ["Does the Explorer use future transactions?", "Can later transactions affect a score?",
           "What is time-safe scoring?", "Is there any look-ahead in the scores?"],
          ["Could a score change because of something that happened later?"]),
    Topic("time-split",
          f"Each address's history is split in time: models are fitted on the earlier {n(M['fitFraction'] * 100)}% of "
          "its transactions and evaluated on the rest, and scalers are fitted inside the same pipeline. No model "
          "learns from the transactions it is evaluated on.",
          [f"{n(M['fitFraction'] * 100)}%", "earlier", "evaluated on the rest"],
          ["How are the models trained and tested?", "What is the time split?",
           "Do models see the transactions they are tested on?", "Why split each history by time?"],
          ["Which part of each history is used for training?"]),
    Topic("feature-size",
          "Size (F1) is how much the transaction moved for the address: what it received minus what it sent, as an "
          "amount in BTC. The Explorer compares it "
          "with the address's usual amount in the same direction, on a logarithmic scale, because amounts range over "
          "many orders of magnitude.",
          ["F1", "BTC", "logarithmic"],
          ["What is the Size feature?", "What does F1 measure?", "What is the amount feature?",
           "How is the transaction amount used?"],
          ["What does a Size reason mean in an explanation?"]),
    Topic("feature-burst",
          f"Burst (F2) counts the address's earlier transactions in the previous {n(F['burstWindowDays'])} days. A "
          "transaction stands out on Burst when that count is far above what is usual for the address, for example "
          "a sudden run of activity from a normally quiet address.",
          ["F2", f"previous {n(F['burstWindowDays'])} days"],
          ["What is the Burst feature?", "What does F2 measure?", "What does 'busier week than usual' mean?",
           "How is a burst of activity detected?"],
          ["What does a Burst reason tell me?"]),
    Topic("feature-gap",
          "Gap (F3) is the time since the address's previous transaction, in hours. A gap much longer than usual can "
          "mean the address woke up after being dormant; a much shorter one means activity came unusually fast.",
          ["F3", "time since", "dormant"],
          ["What is the Gap feature?", "What does F3 measure?", "What does 'longer gap than usual' mean?",
           "Why is the time between transactions a feature?"],
          ["What does a Gap reason tell me?"]),
    Topic("feature-relative-size",
          "Relative size (F4) divides this amount by the address's usual amount in the same direction, so 10x means ten "
          "times the usual. The explanation compares that ratio with the ratio the address usually has, so a higher "
          "ratio than usual can still be below 1x when the address's ratios are usually even lower. Size (F1) says how "
          "big the amount is; Relative size says how big it is for this address.",
          ["F4", "divides", ["ratio the address usually has", "usual ratio"]],
          ["What is the Relative size feature?", "What does F4 measure?", "Why are there two size features?",
           "What does 'higher ratio to the usual amount than usual' mean?"],
          ["Why can a ratio of 0.4x be called higher than usual?", "How is Relative size different from Size?"]),
    Topic("robust-z",
          "A robust z-score says how far a value is from the address's usual value, measured in units of its usual "
          "spread. The Explorer uses the median for the usual value and the median absolute deviation times "
          f"{n(B['madScale'])} for the spread, which a few extreme transactions cannot distort.",
          ["median", "median absolute deviation", n(B["madScale"])],
          ["What is a robust z-score?", "What does z mean in the explanation?", "How is the z-score calculated?",
           "How do you measure how far from normal a value is?"],
          ["Explain the z number shown next to each reason."]),
    Topic("z-of-4",
          "It means the value is about four usual spreads away from the address's usual value. That is well outside "
          f"its normal range and counts as contributing to an explanation (from {n(B['contributingAbsZ'])} upward), but "
          f"the Explorer's flag needs {n(FLAG_Z)} or more on at least one feature.",
          ["contributing", f"{n(FLAG_Z)} or more"],
          ["What does a robust z of 4 mean?", "Is a z of 4 a lot?", "How should I read z = 4?",
           "Would a z of 4 get flagged?"],
          ["My transaction has z = 4 on Size. Is it flagged?"]),
    Topic("why-median",
          "Bitcoin amounts and gaps are extremely skewed: a handful of huge transactions would pull an average and a "
          "standard deviation far away from typical activity. The median and the median absolute deviation ignore "
          "those few extremes, so the usual value stays usual.",
          ["skewed", "median"],
          ["Why use the median instead of the average?", "Why not use mean and standard deviation?",
           "Why is the z-score robust?", "Why not use the average amount as the usual amount?"],
          ["Why do you avoid the mean?"]),
    Topic("spread-floor",
          "If an address is very regular, its spread can be almost zero, and any tiny change would look enormous. The "
          "Explorer therefore uses a minimum spread for each feature, so a small change in a very regular address is "
          "not flagged just because the arithmetic divides by nearly zero.",
          ["minimum spread", ["almost zero", "nearly zero"]],
          ["What happens if an address always sends the same amount?", "Why is there a minimum spread?",
           "What if the usual spread is zero?"],
          ["What stops a tiny change from looking huge for a very regular address?"]),
    Topic("flag-threshold",
          f"With the Explorer's chosen method, a transaction is flagged when at least one feature has a robust z of "
          f"{n(FLAG_Z)} or more in absolute value. Features from {n(B['contributingAbsZ'])} upward are listed as "
          "contributing reasons.",
          [f"{n(FLAG_Z)} or more", "at least one feature", f"{n(B['contributingAbsZ'])} upward"],
          ["When is a transaction flagged?", "What is the flag threshold?", "How unusual does a transaction need to be?",
           "What makes the Explorer flag a transaction?"],
          ["How far from usual does a transaction have to be before it's flagged?"]),
    Topic("why-threshold-5",
          f"The original plan used {n(B['flagAbsZ'])}, but on our data that left {OVER_A1} addresses with more than "
          f"{n(S['sanity']['maxFlaggedShare'] * 100)}% of their transactions flagged. No threshold we tested brought every "
          f"address under that limit. {n(FLAG_Z)} left the fewest ({OVER_OURS}) while "
          f"still finding at least {n(INJ['recallTarget'] * 100)}% of {INJ['patternsMeetingTarget']} of the 3 planted "
          "patterns, so the team chose it. The decision and its evidence are recorded in the change log.",
          [n(B["flagAbsZ"]), "planted"],
          [f"Why is the threshold {n(FLAG_Z)} and not {n(B['flagAbsZ'])}?", "Why was the threshold raised?",
           "Why not use the original threshold?"],
          ["Why did you change the flag threshold from the plan?"]),
    Topic("contributing",
          f"A contributing feature is clearly different from usual (a robust z of {n(B['contributingAbsZ'])} or more) but "
          "not extreme enough to flag the transaction on its own. The explanation lists them so you can see everything "
          "that stands out, strongest first.",
          [f"{n(B['contributingAbsZ'])} or more", "on its own"],
          ["What does 'contributing' mean?", "What is a contributing feature?",
           "Why are some reasons listed but not flagged?"],
          ["Some reasons are listed even though they did not trigger the flag. Why?"]),
    Topic("comparison-window",
          f"The previous {n(F['comparisonWindowDays'])} days of the address's activity, extended further back if needed "
          f"until it holds at least {n(F['comparisonMinTx'])} earlier transactions. Sizes are compared only with earlier "
          "transactions in the same direction (received with received, sent with sent).",
          [f"{n(F['comparisonWindowDays'])} days", f"at least {n(F['comparisonMinTx'])}", "same direction"],
          ["Which earlier transactions are used for the comparison?", "How far back does the Explorer look?",
           "What is the comparison window?", "How much history is used as the baseline?"],
          ["How many days of past activity are compared with a transaction?"]),
    Topic("same-direction",
          "Many addresses receive and send very different amounts, for example many small payments in and a few "
          "large ones out. Comparing a sent amount with received amounts would make ordinary sends look unusual, so "
          "sizes are compared within the same direction.",
          ["same direction", ["very different amounts", "small payments"]],
          ["Why compare received and sent amounts separately?", "What does 'for this direction' mean?",
           "Why does direction matter?", "Why are incoming and outgoing amounts kept apart?"],
          ["Why isn't a send compared with what the address received?"]),
    Topic("not-scored",
          f"A score needs a usual pattern to compare with. An address needs at least {n(F['minHistoryTx'])} earlier "
          f"transactions over at least {n(F['minHistorySpanDays'])} days before its transactions are scored. Earlier "
          "transactions are still shown, just not ranked.",
          [f"at least {n(F['minHistoryTx'])}", f"{n(F['minHistorySpanDays'])} days"],
          ["Why is this transaction not scored?", "What does 'not enough history' mean?",
           "Why can't the Explorer score new addresses?", "How much history does an address need before scoring?"],
          ["Why does this new address show no scores?"]),
    Topic("short-history",
          f"The snapshot includes {DS1['short_addresses']} addresses with only a few transactions on purpose. They show "
          "how the Explorer behaves when there is not enough history: it says so instead of guessing.",
          ["not enough history", "instead of guessing"],
          ["Why are some addresses in the snapshot so short?", "What are the short-history addresses for?",
           "Why include addresses with almost no history?"],
          ["What's the point of the addresses with only a few transactions?"]),
    Topic("reading-explanation",
          "The first line says whether the method flagged the transaction. Then each reason names a feature, says how "
          "this transaction compares with the address's usual value, and gives both values. Start with the first "
          f"reason: it is the strongest. {LIMIT}",
          ["first line", "strongest", LIMIT],
          ["How do I read an explanation?", "What do the lines in the explanation mean?",
           "How should I use the reasons?", "Walk me through an explanation."],
          ["I don't understand the bullet points under a result. What do they mean?"]),
    Topic("what-next",
          "Open the transaction on a block explorer, look at the address's activity around that time, and check "
          "whether the difference has an ordinary reason. Record what you find. A flag is a starting point for that "
          f"review, not its result. {LIMIT}",
          ["block explorer", "ordinary reason", LIMIT],
          ["What should I do with a flagged transaction?", "A transaction is flagged. What now?",
           "How should an analyst follow up a flag?", "What follow-up does a flag call for?"],
          ["I found a flagged transaction. What should my next step be?"]),
    Topic("largest-differences",
          "When no single feature is far from usual, the explanation lists the two features that differ most, so you "
          "can still see how the transaction compares. That is the case for most ordinary transactions, and for a "
          "transaction flagged by a method that looks at the whole pattern at once rather than at one feature.",
          ["two features", "whole pattern"],
          ["Why does a transaction show 'largest differences' instead of reasons?",
           "What does 'largest differences from usual' mean?", "Why is a transaction flagged with no strong reason?"],
          ["The result lists the biggest differences but none of them is very large. Why?"]),
    Topic("methods-list",
          "Nine methods score every transaction: the robust-z baseline; regression of the expected size and gap, used "
          "two ways (how far off the prediction is, and whether it falls outside a calibrated expected range); score "
          "fusion, which combines the baseline and regression; K-Means distance; DBSCAN noise; Isolation Forest; Local "
          "Outlier Factor; and TxGPT, a small language model of transaction sequences. The Explorer shows one of them "
          "by default: the robust-z baseline.",
          ["robust-z baseline", "Isolation Forest", "TxGPT", "K-Means", "by default"],
          ["Which methods does the Explorer use?", "What models are behind the scores?", "How many methods are compared?",
           "List the scoring methods."],
          ["Which algorithms are compared in the Explorer?", "Does the Explorer only use one model?"]),
    Topic("why-baseline",
          "The choice follows a rule fixed in advance: among methods that give stable results when retrained, pick the "
          "one that finds the most planted test transactions, and when two are within five points, the easier one to "
          "explain. Score fusion found slightly more, within that margin, so the baseline won: each of its reasons maps "
          "to one feature.",
          ["stable", "planted", "easier one to explain"],
          ["Why does the Explorer use the baseline method?", "Why was the robust-z baseline chosen?",
           "Why not use the machine-learning methods by default?", "How was the default method chosen?"],
          ["Why isn't the best-scoring machine-learning model the default?"]),
    Topic("planted-transactions",
          "We planted synthetic transactions with known patterns into real histories and measured how many each method "
          f"found: a spike ({n(INJ['spikeMultiplier'])} times the usual amount), a burst ({n(INJ['burstSize'])} "
          f"transactions within {n(INJ['burstWindowHours'])} hours) and a dormancy ({n(INJ['dormantDaysMin'])} to "
          f"{n(INJ['dormantDaysMax'])} days of silence). Real transactions have no labels, so we never claim accuracy on them.",
          ["spike", "burst", "dormancy", "no labels"],
          ["How do you know the methods work without labels?", "What are planted transactions?",
           "How was the Explorer tested?", "How do you evaluate without ground truth?"],
          ["If real transactions have no labels, how do you know a method finds anything?"]),
    Topic("stability",
          "Stability measures whether a method flags the same transactions when it is retrained on slightly different "
          f"data. We require at least {n(M['stabilityTarget'])} overlap between retrained versions; an unstable method's "
          "flags would change from run to run, which an analyst could not rely on.",
          ["retrained", n(M["stabilityTarget"])],
          ["What does stability mean?", "Why does stability matter for choosing a method?", "Why must a method be stable?"],
          ["What happens to a method that flags different transactions every time it's retrained?"]),
    Topic("isolation-forest",
          "Isolation Forest builds many random trees that split transactions apart. Unusual transactions are isolated "
          f"after only a few splits. The Explorer flags the top {n(M['flagTopFraction'] * 100)}% of scores seen during "
          "training.",
          ["random trees", "isolated", f"{n(M['flagTopFraction'] * 100)}%"],
          ["What is Isolation Forest?", "How does Isolation Forest find unusual transactions?",
           "How does an Isolation Forest score a transaction?"],
          ["Explain Isolation Forest in plain words."]),
    Topic("lof",
          "Local Outlier Factor compares how crowded each transaction's neighbourhood is with its neighbours' "
          "neighbourhoods. A transaction in a much sparser spot than its neighbours is unusual.",
          ["neighbourhood", "sparser"],
          ["What is Local Outlier Factor?", "What does LOF do?", "How does LOF decide something is unusual?"],
          ["Explain Local Outlier Factor simply."]),
    Topic("kmeans",
          "K-Means groups transactions into a few typical patterns. A transaction far from every group's centre does "
          "not fit any usual pattern, so its distance to the nearest centre is used as an unusualness score.",
          ["typical patterns", "nearest centre"],
          ["What is K-Means used for?", "How does K-Means distance work?", "What does the K-Means distance measure?"],
          ["Explain the K-Means method in plain words."]),
    Topic("dbscan",
          "DBSCAN finds dense groups of similar transactions and labels points that belong to no dense group as noise. "
          "A transaction with no dense group nearby is flagged.",
          ["dense", "noise"],
          ["What is DBSCAN noise?", "How does DBSCAN flag transactions?", "What does DBSCAN call noise?"],
          ["Explain DBSCAN in plain words."]),
    Topic("regression",
          "Regression models predict the size and the gap a transaction would usually have, given the address's recent "
          "activity. A transaction far from its prediction, or outside its expected range, is unusual. The expected "
          "ranges are calibrated on held-out data so that they contain the stated share of ordinary transactions. Our "
          "tests found regression sensitive but less stable when retrained, so it is not the default.",
          ["predict", "expected range", "less stable"],
          ["What does the regression method do?", "What is the expected range?",
           "How does regression find unusual transactions?", "How are the expected ranges calibrated?"],
          ["Explain the regression method in plain words."]),
    Topic("fusion",
          "Score fusion combines two methods, the robust-z baseline and regression, into one score, and flags the same "
          "share of transactions as the baseline alone. On our planted tests it found slightly more, but the gain was "
          "within the margin the selection rule treats as a tie, so the simpler baseline stays the default.",
          ["baseline", "regression", ["margin", "tie"]],
          ["What is score fusion?", "How does the fusion method work?", "Does combining methods help?",
           "Why isn't fusion the default?"],
          ["What happens when you merge two methods' scores?"]),
    Topic("txgpt",
          "TxGPT is a small language model trained from scratch on transaction sequences instead of words: each "
          "transaction becomes tokens for its gap, direction, amount, inputs, outputs and fee. Its surprisal measures "
          "how unexpected a transaction is given everything before it, and it can say which part was surprising.",
          ["language model", "tokens", "surprisal"],
          ["What is TxGPT?", "How does the transaction language model work?", "What is surprisal?",
           "How does TxGPT turn transactions into tokens?"],
          ["Explain TxGPT in plain words."]),
    Topic("txgpt-training-data",
          "TxGPT learned from thousands of other address histories collected for training, with the addresses stored "
          "only as salted hashes. The addresses the Explorer shows were excluded, so it never scores an address it was "
          "trained on.",
          ["salted hashes", "excluded"],
          ["What did TxGPT learn from?", "Was TxGPT trained on the addresses it scores?",
           "Did TxGPT see the snapshot addresses during training?"],
          ["What data was TxGPT trained on?"]),
    Topic("ds2",
          "DS2 is a separate collection of thousands of address histories from the last year of blocks, with addresses "
          "and transaction ids stored only as salted hashes. It trains TxGPT, and it checks that the methods chosen on "
          "the snapshot still work on many more addresses.",
          ["salted hashes", "TxGPT", "many more addresses"],
          ["What is DS2?", "What is the large crawl used for?", "Why collect thousands of extra addresses?"],
          ["What is the second Esplora collection for?"]),
    Topic("ds10",
          "A teammate collected a second, independent snapshot by the same rule, from addresses no model had seen. "
          "Every method was tested on it frozen, exactly as trained. TxGPT found the most planted patterns there and "
          "the baseline came second; bursts were harder to find because those addresses were busier.",
          ["independent", "frozen", "TxGPT"],
          ["What is DS10?", "Was the Explorer tested on new data?", "How well do the methods work on unseen addresses?",
           "What did the teammate's snapshot show?"],
          ["Did you check the methods on addresses they had never seen?"]),
    Topic("assistant",
          "Every explanation starts as a fixed template filled with the transaction's numbers. Our language models may "
          "rephrase it, but code checks every rephrased answer: all numbers must match the facts, comparison words "
          "must agree with the data, flagged items must keep the limitation line, and some words are not allowed. If "
          "any check fails, the template is shown instead.",
          ["template", "code checks", "numbers must match"],
          ["Where do the explanation texts come from?", "Does a language model write the explanations?",
           "How is an explanation produced?", "Is the explanation text fixed or generated?"],
          # Round 1's frozen test questions on this topic (data/processed/ds7/test.jsonl).
          ["Who writes the explanations?", "Is the explanation written by AI?", "How are the explanations generated?"]),
    Topic("guardrail",
          "Its answer is not shown. A code check compares every number in the answer with the transaction's facts, and "
          "every comparison word (larger, quieter, longer and so on) with the direction of the data. One unknown number, "
          "a wrong comparison word, a missing limitation line or a disallowed word is enough to fall back to the "
          "template, which is always correct.",
          ["every number", "comparison word", "template"],
          ["What happens if the AI gets a number wrong?", "What is the guardrail?", "Can the assistant make things up?",
           "What checks does an AI answer have to pass?"],
          ["What if the model says a week was quieter when it was busier?"]),
    Topic("two-assistants",
          "By default the Explorer shows the template. When the Assistant is switched on, two models can rephrase it. A "
          "pretrained 3-billion-parameter model (Qwen2.5-3B-Instruct), fine-tuned by us, writes first where a GPU is "
          "available and also answers general questions. A 152-million-parameter model, trained from scratch by our "
          "team, runs on an ordinary CPU and writes next. Both go through the same checks, and the template is the "
          "final fallback.",
          ["152-million", ["Qwen", "3-billion"], "from scratch", "template"],
          ["Why are there two language models in the Explorer?",
           "What is the difference between the small and the large model?", "What is Qwen used for?",
           "Which model runs without a GPU?"],
          ["Why do you have a small model and a big model?"]),
    Topic("labelled-datasets",
          "Two public labelled datasets: Elliptic, Bitcoin transactions labelled by the dataset's authors, and "
          "BitcoinHeist, address-days labelled with ransomware families. We use them to train and test classifiers "
          "with accuracy, precision, recall and F1, and to check how far unusual coincides with their labels. They are "
          "never used to label the Explorer's addresses.",
          ["Elliptic", "BitcoinHeist", "precision"],
          ["Which labelled datasets did you use?", "What are Elliptic and BitcoinHeist used for?",
           "Where were the classifiers trained?"],
          ["Which datasets have labels, and what are they used for?"]),
    Topic("label-check",
          "Our unusualness methods ranked the labelled positive transactions below chance on both datasets, while a "
          "classifier trained on the labels did far better. In other words, labelled activity often looks typical. "
          "That is why the Explorer only ever says unusual.",
          ["below chance", "classifier"],
          ["Does unusual mean the same as labelled in Elliptic?", "What did the label check find?",
           "Do unusual transactions match the datasets' labels?"],
          ["Are the most unusual transactions the labelled ones?"]),
    Topic("drift",
          "Patterns change over time. On Elliptic our best model scored well until a point late in the test period and "
          "then dropped for every model, a change the dataset's authors also report. Testing on later data than the "
          "training data is what reveals this.",
          ["change over time", "Elliptic", "later data"],
          ["Why does a model trained on early data fail later?", "What happened to the Elliptic results late in the test period?",
           "Why test on later data than the training data?"],
          # Round 1's frozen test questions on this topic.
          ["What is drift?", "Why did the classifiers get worse on later data?"]),
    Topic("privacy",
          "The Explorer uses only public blockchain data and never links addresses to people. The large training "
          "collection stores addresses and transaction ids only as salted hashes, and the salt is never shared.",
          ["public blockchain data", "salted hashes"],
          ["Do you store personal data?", "Is this tool private?", "How are addresses protected?"],
          ["Does the tool keep any personal information?"]),
    Topic("btc-units",
          "Amounts are shown in BTC. Internally they are counted in satoshis: one bitcoin is 100,000,000 satoshis, so no "
          "rounding error builds up.",
          ["BTC", "100,000,000"],
          ["What units are the amounts in?", "What is a satoshi?", "How many satoshis are in one bitcoin?"],
          ["Are amounts shown in bitcoin or satoshis?"]),
    Topic("direction",
          "For each transaction the Explorer adds up what the address received and what it sent. If it received more, "
          "the direction is in (received); if it sent more, it is out (sent).",
          ["received more", "sent more"],
          ["What do 'received' and 'sent' mean for an address?", "How is direction decided?",
           "What does direction 'in' mean?"],
          ["How do you tell whether a transaction was incoming or outgoing?"]),
    Topic("timezone",
          "All times are in UTC, taken from the block that confirmed the transaction.",
          ["UTC", "block"],
          ["What time zone are the times in?", "Are times local?", "Which clock are timestamps in?"],
          ["Are the transaction times in my local time?"]),
    Topic("fee-rate",
          "The fee rate is the fee divided by the transaction's size, in satoshis per virtual byte. TxGPT uses it as "
          "part of a transaction's shape; the main explanations do not.",
          ["satoshis per virtual byte", "TxGPT"],
          ["What is the fee rate?", "Why does the fee matter?", "How is the fee rate calculated?"],
          ["Do transaction fees affect the flags?"]),
    Topic("inputs-outputs",
          "Inputs are the earlier coins a transaction spends; outputs are where the value goes. Their counts describe the "
          "transaction's shape, for example many inputs when a wallet gathers small coins together.",
          ["earlier coins", "where the value goes", "shape"],
          ["What are inputs and outputs?", "Why does the number of inputs matter?",
           "What does the number of outputs tell me?"],
          ["Why would a transaction have many inputs?"]),
    Topic("what-if",
          "You can enter a hypothetical amount, direction and time for one of the snapshot's addresses. The Explorer "
          "scores it against that address's real history before that time and explains the result the same way.",
          ["hypothetical", "real history"],
          ["What is the what-if check?", "Can I test a hypothetical transaction?", "How does the what-if feature work?"],
          ["Can I check how unusual an imaginary payment would be?"]),
    Topic("flag-rate",
          "It varies by address. Most addresses have few flags, but an address whose behaviour changed, for example "
          "one that became much busier, can have many in that period. A run of flags usually means the address changed "
          "its pattern, which is itself worth a look.",
          ["varies by address", "changed its pattern"],
          ["How many transactions are flagged?", "Is it normal that many transactions are flagged?",
           "Why are flag counts so different between addresses?"],
          ["Why does one address have so many more flags than another?"]),
    Topic("behaviour-change",
          "Flags that cluster in one period usually mean the address's behaviour changed then: it became busier, "
          "started sending instead of receiving, or broke a very regular routine. Look at the first flagged transaction "
          f"of the run; the later ones are mostly the same change. {LIMIT}",
          ["behaviour changed", "first flagged transaction", LIMIT],
          ["Why are so many of this address's transactions flagged?", "What does it mean when flags cluster?",
           "Why do flags come in runs?"],
          ["Lots of flags in a row on one address. What does that mean?"]),
    Topic("false-alarms",
          "Yes. A flag measures difference from the past, and ordinary events cause differences too. Short histories "
          "give less reliable baselines, and a public snapshot cannot show off-chain context. The Explorer is a way to "
          "prioritise review, not a replacement for it.",
          ["difference from the past", "short histories", "prioritise review"],
          ["Can the Explorer be wrong?", "What are the limitations of the Explorer?", "Can a normal transaction be flagged?"],
          ["How reliable is a flag?"]),
    Topic("missed",
          "Yes. A transaction can be similar to the address's past and still matter, and our planted tests show every "
          "method misses some patterns, bursts especially. No flag means nothing stands out against this address's "
          "own history, nothing more.",
          ["bursts", "nothing stands out"],
          ["Can the Explorer miss something?", "Does no flag mean everything is fine?",
           "Are there patterns the methods miss?"],
          ["If nothing is flagged, can I assume the address is fine?"]),
    Topic("why-these-addresses",
          f"By a rule written down before any data was fetched: take the addresses paid in {PICK['snapshotBlockCount']} "
          "evenly spaced blocks from the four weeks before the snapshot, visit them in a shuffled order, and keep "
          f"addresses with {PICK['mainMinTx']} to {PICK['mainMaxTx']:,} transactions over at least "
          f"{PICK['mainMinSpanDays']} days. The upper limit leaves out the busiest services such as exchanges. The "
          "rule and every skipped address are recorded in the snapshot's manifest.",
          ["rule written down", "shuffled", "manifest"],
          ["How were the addresses chosen?", f"Why these {DS1['main_addresses'] + DS1['short_addresses']} addresses?",
           "How did you pick the snapshot addresses?"],
          ["Were the addresses hand-picked?"]),
    Topic("reproducible",
          "Yes. One command rebuilds the features, the scoring models and every result table from the saved data, and "
          "every number in our report is read from tables the code writes. The README lists the commands.",
          ["one command", "README"],
          ["Can I reproduce these results?", "How do I rerun the analysis?", "Can someone else rebuild your results?"],
          ["How can I check your numbers myself?"]),
    Topic("cybersecurity",
          "Investigations often start from activity that departs from an established pattern, for example a wallet "
          "suddenly moving far more than usual. The Explorer surfaces those departures quickly and explains them, so "
          "a person can decide what deserves attention.",
          ["established pattern", "decide what deserves attention"],
          ["How is this a cybersecurity tool?", "Why does unusual activity matter for security?",
           "Where does the Explorer fit in an investigation?"],
          ["What does this have to do with security?"]),
    Topic("chat-scope",
          "I can explain how the Explorer works, what its features, methods and flags mean, and how to read an "
          "explanation. I can't say who controls an address or whether anyone did anything wrong.",
          ["how the Explorer works", "who controls an address"],
          ["What can you help me with?", "What kind of questions can I ask you?", "What are you able to answer?"],
          ["What topics can I ask you about?"]),
    Topic("dont-know",
          "I can't answer that, and the Explorer is not designed to. It only compares transactions with an address's own "
          f"history and cannot say who owns an address or what they intended. {LIMIT}",
          ["can't answer", "own history", LIMIT],
          ["Can you tell me if this person is a scammer?", "Is this wallet owned by a hacker?",
           "Tell me which of these addresses belongs to a thief.", "Which of these wallets is run by a gang?"],
          ["Just tell me: is this wallet a scam or not?"]),
]


def facts_in(answer: str, key_facts: list[str | list[str]]) -> list[str | list[str]]:
    """The key facts the reference answer itself does not state (must be none)."""
    text = " ".join(answer.lower().split())
    return [f for f in key_facts if not any(" ".join(p.lower().split()) in text for p in ([f] if isinstance(f, str) else f))]


def main() -> int:
    banned = re.compile(r"\b(" + "|".join(map(re.escape, S["bannedWords"])) + r")", re.IGNORECASE)
    rows, problems, seen = [], [], set()
    for i, t in enumerate(TOPICS):
        if banned.search(t.answer):
            problems.append(f"{t.name}: banned word in the answer")
        if t.name in NEEDS_LIMIT and LIMIT not in t.answer:
            problems.append(f"{t.name}: the limitation line is required")
        if missing := facts_in(t.answer, t.key_facts):
            problems.append(f"{t.name}: key facts missing from its own answer: {missing}")
        if not t.held_out or len(t.taught) < 3:
            problems.append(f"{t.name}: needs at least 3 taught and 1 held-out phrasing")
        for q in t.taught + t.held_out:
            if q in seen:
                problems.append(f"{t.name}: question used twice: {q!r}")
            seen.add(q)
        splits = ["train"] * len(t.taught)
        if i % VAL_EVERY == 0:
            splits[-1] = "val"
        rows += [{"topic": t.name, "question": q, "answer": t.answer, "split": s, "status": STATUS}
                 for q, s in zip(t.taught, splits)]
        rows += [{"topic": t.name, "question": q, "answer": t.answer, "split": "test", "key_facts": t.key_facts,
                  "status": STATUS} for q in t.held_out]
    if problems:
        print("\n".join(problems))
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"Wrote {len(rows)} pairs on {len(TOPICS)} topics to {OUT}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
