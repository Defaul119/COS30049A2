"""Stage D6 (explorer/ensemble.py): the ensemble search and its paired comparison."""
from __future__ import annotations

import numpy as np

from explorer import ensemble


def members(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Four members' probabilities: the third is good, the fourth noisier, the first two noise."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    good = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.15, n), 0, 1)
    fair = np.clip(0.3 * y + 0.35 + rng.normal(0, 0.2, n), 0, 1)
    return np.column_stack([rng.random(n), rng.random(n), good, fair]), y


def test_search_needs_two_members_and_drops_duplicate_votes():
    p, y = members(400, 0)
    table, winner = ensemble.search(p, y)
    voting = table[table["combine"] == "mean"]
    assert (voting["members"] >= 2).all()
    weights = voting[[f"w_{k}" for k in ensemble.MODEL_KINDS]].to_numpy()
    shares = {tuple(np.round(w / w.sum(), 9)) for w in weights}
    assert len(shares) == len(voting)  # no two candidates cast the same vote
    assert table.loc[0, "chosen"] and table.loc[0, "f1"] == table["f1"].max()
    assert winner.combine == table.loc[0, "combine"]


def test_ensemble_scores_are_probabilities():
    p, y = members(300, 1)
    _, winner = ensemble.search(p, y)
    s = winner.score(p)
    assert s.shape == (300,) and (s >= 0).all() and (s <= 1).all()


def test_compare_finds_a_clear_gain():
    p, y = members(2000, 2)
    better, worse = p[:, 2], p[:, 3]
    c = ensemble.compare(y, better, worse, (0.5, 0.5))
    assert c["diff"] > 0 and c["diff_ci_low"] > 0 and c["mcnemar_p"] < 0.05
    same = ensemble.compare(y, better, better, (0.5, 0.5))
    assert same["diff"] == 0 and same["mcnemar_p"] == 1.0
