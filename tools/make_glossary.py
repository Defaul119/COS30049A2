"""Write data/processed/ds7/glossary.jsonl: the DS7 glossary pairs (plan §6.7, "glossary pairs").

The plan says the team writes about 300 questions and answers about the Explorer. On 23 Oct 2026
the team delegated this to Cao Tri Tran, and Claude drafted them (declared in docs/AI_USE_LOG.md)
for review. Each topic has one answer and several phrasings of the question; explorer.ds7 splits
by topic, so a paraphrase of a training question never reaches the test set.

Every number in an answer is read from config/settings.json or models/baseline.json when this
script runs, so the glossary cannot drift from the code. The script refuses to write if an
answer uses a banned word, or misses the limitation line where one is required.

Usage:  python tools/make_glossary.py      (then: python -m explorer.cli build-ds7)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "config" / "settings.json").read_text(encoding="utf-8"))
FLAG_Z = json.loads((ROOT / "models" / "baseline.json").read_text(encoding="utf-8"))["chosen"]["refined"]["threshold"]
F, B, M, INJ = S["features"], S["baseline"], S["model"], S["injection"]
LIMIT = S["limitationLine"]
OUT = ROOT / "data" / "processed" / "ds7" / "glossary.jsonl"


def n(x: float) -> str:
    """A number as the answers print it: 5.0 -> 5, 0.02 -> 0.02."""
    return f"{x:g}"


# (topic, [question phrasings], answer)
TOPICS: list[tuple[str, list[str], str]] = [
    ("what-is-explorer",
     ["What is the Bitcoin Address Activity Explorer?", "What does this tool do?", "What is the Explorer for?",
      "Can you explain what this app is?"],
     "The Explorer shows a Bitcoin address's transaction history and ranks each transaction by how unusual it is "
     "compared with that same address's own past. For each flagged transaction it gives a plain-language reason. "
     "It is an investigation aid that points to transactions worth a look; it does not judge anyone."),
    ("intended-users",
     ["Who is the Explorer for?", "Who are the intended users?", "Who should use this tool?"],
     "It is for analysts and investigators who need to review an address's activity quickly, for example a "
     "compliance analyst or a researcher. It helps them decide where to look first; the judgement stays with them."),
    ("not-a-judgement",
     ["Does a flag mean the owner did something wrong?", "Is a flagged transaction proof of wrongdoing?",
      "Is this address a criminal?", "Does this mean the address is involved in fraud?"],
     "No. A flag only means the transaction is unusual for this address's own history: larger, faster, busier or "
     "differently shaped than usual. There are many ordinary reasons for that, such as a one-off purchase or a "
     f"change of wallet software. {LIMIT}"),
    ("who-owns-address",
     ["Who owns this address?", "Can you tell me whose wallet this is?", "Can the Explorer identify the person behind an address?"],
     "No. The Explorer only uses public blockchain data: amounts, times and the shape of transactions. It never "
     "says who controls an address or why it transacts, and it should not be used to guess."),
    ("why-unusual-not-wrongdoing",
     ["Why don't you call unusual transactions bad?", "Why is unusual not the same as wrongdoing?",
      "Why does the Explorer avoid accusing anyone?"],
     "Because the evidence says they are different things. When we scored two labelled public datasets with our "
     "unusualness methods, the labelled positive transactions were, if anything, less unusual than the rest. "
     f"Unusual is a reason to look, not a conclusion. {LIMIT}"),
    ("limitation-line",
     ["What does the line at the end of a flag mean?", "Why does every flag end with the same sentence?",
      "What is the limitation line?"],
     f"Every flagged item ends with: \"{LIMIT}\" It is there so that nobody reads a flag as a verdict. The Explorer "
     "measures how different a transaction is from the address's past, nothing more."),
    ("data-source",
     ["Where does the data come from?", "What data does the Explorer use?", "Which API is the data from?"],
     "From public Bitcoin blockchain data, fetched through the Esplora API (Blockstream, with mempool.space as a "
     "fallback when Blockstream rate-limits). The Explorer works on a saved snapshot, so results can be reproduced."),
    ("snapshot",
     ["Why is the data a snapshot?", "Is the data live?", "Does the Explorer show the latest transactions?"],
     "It uses a fixed snapshot of 25 addresses with long histories and 3 with short ones, saved once with "
     "checksums. A snapshot makes every result reproducible and avoids depending on a public API's limits. New "
     "transactions after the snapshot are not shown."),
    ("own-history",
     ["Compared with what is a transaction unusual?", "What is the transaction compared against?",
      "Why compare an address only with itself?"],
     "With the same address's own earlier transactions. Addresses differ hugely: in our snapshot the typical "
     "amount of one address is thousands of times that of another, so one global threshold would flag the big "
     "addresses all the time and the small ones never."),
    ("strictly-earlier",
     ["Does the Explorer use future transactions?", "Can later transactions affect a score?", "What is time-safe scoring?"],
     "No. A transaction is compared only with transactions that happened strictly before it. Transactions in the "
     "same block are never each other's history. This keeps every score something an analyst could have "
     "computed at that moment."),
    ("feature-size",
     ["What is the Size feature?", "What does F1 measure?", "What is the amount feature?"],
     "Size (F1) is how much the address received or sent in the transaction, in BTC. The Explorer compares it "
     "with the address's usual amount in the same direction, on a logarithmic scale, because amounts range over "
     "many orders of magnitude."),
    ("feature-burst",
     ["What is the Burst feature?", "What does F2 measure?", "What does 'busier week than usual' mean?"],
     f"Burst (F2) counts the address's earlier transactions in the previous {n(F['burstWindowDays'])} days. A "
     "transaction is flagged on Burst when that count is far above what is usual for the address, for example "
     "a sudden run of activity from a normally quiet address."),
    ("feature-gap",
     ["What is the Gap feature?", "What does F3 measure?", "What does 'longer gap than usual' mean?"],
     "Gap (F3) is the time since the address's previous transaction, in hours. A gap much longer than usual can "
     "mean the address woke up after being dormant; a much shorter one means activity came unusually fast."),
    ("feature-relative-size",
     ["What is the Relative size feature?", "What does F4 measure?", "Why are there two size features?"],
     "Relative size (F4) divides this amount by the address's usual amount in the same direction, so 10x means ten "
     "times the usual. Size (F1) says how big the amount is; Relative size says how big it is for this address. "
     "Our analysis found they are only moderately related, so both are kept."),
    ("robust-z",
     ["What is a robust z-score?", "What does z mean in the explanation?", "How is the z-score calculated?"],
     "A robust z-score says how far a value is from the address's usual value, measured in units of its usual "
     f"spread. The Explorer uses the median for the usual value and the median absolute deviation times "
     f"{n(B['madScale'])} for the spread, which a few extreme transactions cannot distort."),
    ("z-of-4",
     ["What does a robust z of 4 mean?", "Is a z of 4 a lot?", "How should I read z = 4?"],
     "It means the value is about four usual spreads away from the address's usual value. That is well outside "
     f"its normal range and counts as contributing to an explanation (from {n(B['contributingAbsZ'])} upward), but "
     f"the Explorer's flag needs {n(FLAG_Z)} or more on at least one feature."),
    ("why-median",
     ["Why use the median instead of the average?", "Why not use mean and standard deviation?",
      "Why is the z-score robust?"],
     "Bitcoin amounts and gaps are extremely skewed: a handful of huge transactions would pull an average and a "
     "standard deviation far away from typical activity. The median and the median absolute deviation ignore "
     "those few extremes, so the usual value stays usual."),
    ("spread-floor",
     ["What happens if an address always sends the same amount?", "Why is there a minimum spread?",
      "What if the usual spread is zero?"],
     "If an address is very regular, its spread can be almost zero, and any tiny change would look enormous. The "
     "Explorer therefore uses a minimum spread for each feature, so a small change in a very regular address is "
     "not flagged just because the arithmetic divides by nearly zero."),
    ("flag-threshold",
     ["When is a transaction flagged?", "What is the flag threshold?", "How unusual does a transaction need to be?"],
     f"With the Explorer's chosen method, a transaction is flagged when at least one feature has a robust z of "
     f"{n(FLAG_Z)} or more in absolute value. Features from {n(B['contributingAbsZ'])} upward are listed as "
     "contributing reasons."),
    ("why-threshold-5",
     ["Why is the threshold 5 and not 3.5?", "Why was the threshold raised?", "Why not use the original threshold?"],
     f"The original plan used 3.5, but on our data it flagged far more than intended and several addresses had "
     f"too many flags. Testing thresholds on planted transactions, {n(FLAG_Z)} gave the fewest addresses over the "
     "limit while still finding most planted patterns. The decision and its evidence are recorded in the change log."),
    ("contributing",
     ["What does 'contributing' mean?", "What is a contributing feature?", "Why are some reasons listed but not flagged?"],
     f"A contributing feature is clearly different from usual (a robust z of {n(B['contributingAbsZ'])} or more) but "
     "not extreme enough to flag the transaction on its own. The explanation lists them so you can see everything "
     "that stands out, strongest first."),
    ("comparison-window",
     ["Which earlier transactions are used for the comparison?", "How far back does the Explorer look?",
      "What is the comparison window?"],
     f"The previous {n(F['comparisonWindowDays'])} days of the address's activity, extended further back if needed "
     f"until it holds at least {n(F['comparisonMinTx'])} earlier transactions. Sizes are compared only with earlier "
     "transactions in the same direction (received with received, sent with sent)."),
    ("same-direction",
     ["Why compare received and sent amounts separately?", "What does 'for this direction' mean?",
      "Why does direction matter?"],
     "Many addresses receive and send very different amounts, for example many small payments in and a few "
     "large ones out. Comparing a sent amount with received amounts would make ordinary sends look unusual, so "
     "sizes are compared within the same direction."),
    ("not-scored",
     ["Why is this transaction not scored?", "What does 'not enough history' mean?", "Why can't the Explorer score new addresses?"],
     f"A score needs a usual pattern to compare with. An address needs at least {n(F['minHistoryTx'])} earlier "
     f"transactions over at least {n(F['minHistorySpanDays'])} days before its transactions are scored. Earlier "
     "transactions are still shown, just not ranked."),
    ("short-history",
     ["Why are some addresses in the snapshot so short?", "What are the short-history addresses for?"],
     "The snapshot includes three addresses with only a few transactions on purpose. They show how the Explorer "
     "behaves when there is not enough history: it says so instead of guessing."),
    ("reading-explanation",
     ["How do I read an explanation?", "What do the lines in the explanation mean?", "How should I use the reasons?"],
     "The first line says whether the method flagged the transaction. Then each reason names a feature, says how "
     "this transaction compares with the address's usual value, and gives both values. Start with the first "
     f"reason: it is the strongest. {LIMIT}"),
    ("what-next",
     ["What should I do with a flagged transaction?", "A transaction is flagged. What now?",
      "How should an analyst follow up a flag?"],
     "Open the transaction on a block explorer, look at the address's activity around that time, and check "
     "whether the difference has an ordinary reason. Record what you find. A flag is a starting point for that "
     f"review, not its result. {LIMIT}"),
    ("largest-differences",
     ["Why does a transaction show 'largest differences' instead of reasons?",
      "What does 'largest differences from usual' mean?"],
     "When no single feature is far from usual, the Explorer still lists the two features that differ most, so "
     "you can see what the method reacted to. This happens with methods that look at the whole pattern at once."),
    ("methods-list",
     ["Which methods does the Explorer use?", "What models are behind the scores?", "How many methods are compared?"],
     "Eight methods score every transaction: the robust-z baseline; regression of the expected size and gap, used "
     "two ways (how far off the prediction is, and whether it falls outside an expected range); K-Means distance; "
     "DBSCAN noise; Isolation Forest; Local Outlier Factor; and TxGPT, a small language model of transaction "
     "sequences. The Explorer shows one of them by default: the robust-z baseline."),
    ("why-baseline",
     ["Why does the Explorer use the baseline method?", "Why was the robust-z baseline chosen?",
      "Why not use the machine-learning methods by default?"],
     "The choice follows a rule fixed in advance: among methods that give stable results when retrained, pick the "
     "one that finds the most planted test transactions. The baseline won on that rule, and it is also the "
     "easiest to explain, because each reason maps to one feature."),
    ("planted-transactions",
     ["How do you know the methods work without labels?", "What are planted transactions?", "How was the Explorer tested?"],
     "We planted synthetic transactions with known patterns into real histories and measured how many each method "
     f"found: a spike ({n(INJ['spikeMultiplier'])} times the usual amount), a burst ({n(INJ['burstSize'])} "
     f"transactions within {n(INJ['burstWindowHours'])} hours) and a dormancy ({n(INJ['dormantDaysMin'])} to "
     f"{n(INJ['dormantDaysMax'])} days of silence). Real transactions have no labels, so we never claim accuracy on them."),
    ("stability",
     ["What does stability mean?", "Why does stability matter for choosing a method?"],
     "Stability measures whether a method flags the same transactions when it is retrained on slightly different "
     f"data. We require at least {n(M['stabilityTarget'])} overlap between retrained versions; an unstable method's "
     "flags would change from run to run, which an analyst could not rely on."),
    ("isolation-forest",
     ["What is Isolation Forest?", "How does Isolation Forest find unusual transactions?"],
     "Isolation Forest builds many random trees that split transactions apart. Unusual transactions are isolated "
     f"after only a few splits. The Explorer flags the top {n(M['flagTopFraction'] * 100)}% of scores seen during "
     "training."),
    ("lof",
     ["What is Local Outlier Factor?", "What does LOF do?"],
     "Local Outlier Factor compares how crowded each transaction's neighbourhood is with its neighbours' "
     "neighbourhoods. A transaction in a much sparser spot than its neighbours is unusual."),
    ("kmeans",
     ["What is K-Means used for?", "How does K-Means distance work?"],
     "K-Means groups transactions into a few typical patterns. A transaction far from every group's centre does "
     "not fit any usual pattern, so its distance to the nearest centre is used as an unusualness score."),
    ("dbscan",
     ["What is DBSCAN noise?", "How does DBSCAN flag transactions?"],
     "DBSCAN finds dense groups of similar transactions and labels points that belong to no dense group as noise. "
     "A transaction with no dense group nearby is flagged."),
    ("regression",
     ["What does the regression method do?", "What is the expected range?", "How does regression find unusual transactions?"],
     "Regression models predict the size and the gap a transaction would usually have, given the address's recent "
     "activity. A transaction far outside its predicted range is unusual. Our tests found this method sensitive "
     "but less stable when retrained, so it is not the default."),
    ("txgpt",
     ["What is TxGPT?", "How does the transaction language model work?", "What is surprisal?"],
     "TxGPT is a small language model trained from scratch on transaction sequences instead of words: each "
     "transaction becomes tokens for its gap, direction, amount, inputs, outputs and fee. Its surprisal measures "
     "how unexpected a transaction is given everything before it, and it can say which part was surprising."),
    ("txgpt-training-data",
     ["What did TxGPT learn from?", "Was TxGPT trained on the addresses it scores?"],
     "TxGPT learned from thousands of other address histories collected for training, with the addresses stored "
     "only as salted hashes. The addresses the Explorer shows were excluded, so it never scores an address it was "
     "trained on."),
    ("assistant",
     ["Who writes the explanations?", "Is the explanation written by AI?", "How are the explanations generated?"],
     "Every explanation starts as a fixed template filled with the transaction's numbers. Our language models may "
     "rephrase it, but code checks every rephrased answer: all numbers must match the facts, flagged items must "
     "keep the limitation line, and some words are not allowed. If any check fails, the template is shown instead."),
    ("guardrail",
     ["What happens if the AI gets a number wrong?", "What is the guardrail?", "Can the assistant make things up?"],
     "Its answer is not shown. A code check compares every number in the answer with the transaction's facts; one "
     "unknown number, a missing limitation line or a disallowed word is enough to fall back to the template, which "
     "is always correct."),
    ("two-assistants",
     ["Why are there two language models in the Explorer?", "What is the difference between the small and the large model?"],
     "A 152-million-parameter model, trained from scratch by our team, writes the first explanation. A larger "
     "pretrained model, fine-tuned by us, is the second writer and also answers general questions. Both go through "
     "the same checks, and the template is the final fallback."),
    ("labelled-datasets",
     ["Which labelled datasets did you use?", "What are Elliptic and BitcoinHeist used for?"],
     "Two public labelled datasets: Elliptic, Bitcoin transactions labelled by the dataset's authors, and "
     "BitcoinHeist, address-days labelled with ransomware families. We use them to train and test classifiers "
     "with accuracy, precision, recall and F1, and to check how far unusual coincides with their labels. They are "
     "never used to label the Explorer's addresses."),
    ("label-check",
     ["Does unusual mean the same as labelled in Elliptic?", "What did the label check find?"],
     "Our unusualness methods ranked the labelled positive transactions below chance on both datasets, while a "
     "classifier trained on the labels did far better. In other words, labelled activity often looks typical. "
     "That is why the Explorer only ever says unusual."),
    ("drift",
     ["Why did the classifiers get worse on later data?", "What is drift?"],
     "Patterns change over time. On Elliptic our best model scored well until a point late in the test period and "
     "then dropped for every model, a change the dataset's authors also report. Testing on later data than the "
     "training data is what reveals this."),
    ("privacy",
     ["Do you store personal data?", "Is this tool private?", "How are addresses protected?"],
     "The Explorer uses only public blockchain data and never links addresses to people. The large training "
     "collection stores addresses and transaction ids only as salted hashes, and the salt is never shared."),
    ("btc-units",
     ["What units are the amounts in?", "What is a satoshi?"],
     "Amounts are shown in BTC. Internally they are counted in satoshis: one bitcoin is 100,000,000 satoshis, so no "
     "rounding error builds up."),
    ("direction",
     ["What do 'received' and 'sent' mean for an address?", "How is direction decided?"],
     "For each transaction the Explorer adds up what the address received and what it sent. If it received more, "
     "the direction is in (received); if it sent more, it is out (sent)."),
    ("timezone",
     ["What time zone are the times in?", "Are times local?"],
     "All times are in UTC, taken from the block that confirmed the transaction."),
    ("fee-rate",
     ["What is the fee rate?", "Why does the fee matter?"],
     "The fee rate is the fee divided by the transaction's size, in satoshis per virtual byte. TxGPT uses it as "
     "part of a transaction's shape; the main explanations do not."),
    ("inputs-outputs",
     ["What are inputs and outputs?", "Why does the number of inputs matter?"],
     "Inputs are the earlier coins a transaction spends; outputs are where the value goes. Their counts describe the "
     "transaction's shape, for example many inputs when a wallet gathers small coins together."),
    ("what-if",
     ["What is the what-if check?", "Can I test a hypothetical transaction?"],
     "You can enter a hypothetical amount, direction and time for one of the snapshot's addresses. The Explorer "
     "scores it against that address's real history before that time and explains the result the same way."),
    ("flag-rate",
     ["How many transactions are flagged?", "Is it normal that many transactions are flagged?"],
     "It varies by address. Most addresses have few flags, but an address whose behaviour changed, for example "
     "one that became much busier, can have many in that period. A run of flags usually means the address changed "
     "its pattern, which is itself worth a look."),
    ("behaviour-change",
     ["Why are so many of this address's transactions flagged?", "What does it mean when flags cluster?"],
     "Flags that cluster in one period usually mean the address's behaviour changed then: it became busier, "
     "started sending instead of receiving, or broke a very regular routine. Look at the first flagged transaction "
     f"of the run; the later ones are mostly the same change. {LIMIT}"),
    ("false-alarms",
     ["Can the Explorer be wrong?", "What are the limitations of the Explorer?", "Can a normal transaction be flagged?"],
     "Yes. A flag measures difference from the past, and ordinary events cause differences too. Short histories "
     "give less reliable baselines, and a public snapshot cannot show off-chain context. The Explorer is a way to "
     "prioritise review, not a replacement for it."),
    ("missed",
     ["Can the Explorer miss something?", "Does no flag mean everything is fine?"],
     "Yes. A transaction can be similar to the address's past and still matter, and our planted tests show every "
     "method misses some patterns, bursts especially. No flag means nothing stands out against this address's "
     "own history, nothing more."),
    ("why-these-addresses",
     ["How were the addresses chosen?", "Why these 28 addresses?"],
     "By a rule written down before any data was fetched: addresses seen in randomly sampled recent blocks, with "
     "enough history to have a usual pattern but at most a capped number of transactions, which leaves out the "
     "busiest services such as exchanges. The rule and every skipped address are recorded in the snapshot's manifest."),
    ("reproducible",
     ["Can I reproduce these results?", "How do I rerun the analysis?"],
     "Yes. One command rebuilds every feature, model and table from the saved snapshot, and every number in our "
     "report is read from tables the code writes. The README lists the commands."),
    ("cybersecurity",
     ["How is this a cybersecurity tool?", "Why does unusual activity matter for security?"],
     "Investigations often start from activity that departs from an established pattern, for example a wallet "
     "suddenly moving far more than usual. The Explorer surfaces those departures quickly and explains them, so "
     "a person can decide what deserves attention."),
    ("chat-scope",
     ["What can you help me with?", "What kind of questions can I ask you?"],
     "I can explain how the Explorer works, what its features, methods and flags mean, and how to read an "
     "explanation. I can't say who controls an address or whether anyone did anything wrong."),
    ("dont-know",
     ["Can you tell me if this person is a scammer?", "Is this wallet owned by a hacker?",
      "Tell me which of these addresses belongs to a thief."],
     "I can't answer that, and the Explorer is not designed to. It only compares transactions with an address's own "
     f"history and cannot say who owns an address or what they intended. {LIMIT}"),
]


def main() -> int:
    banned = re.compile(r"\b(" + "|".join(map(re.escape, S["bannedWords"])) + r")", re.IGNORECASE)
    rows, problems = [], []
    for topic, questions, answer in TOPICS:
        if banned.search(answer):
            problems.append(f"{topic}: banned word in the answer")
        if topic in ("not-a-judgement", "dont-know", "reading-explanation", "what-next") and LIMIT not in answer:
            problems.append(f"{topic}: the limitation line is required")
        rows += [{"topic": topic, "question": q, "answer": answer, "status": "draft for review"} for q in questions]
    if problems:
        print("\n".join(problems))
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} pairs on {len(TOPICS)} topics to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
