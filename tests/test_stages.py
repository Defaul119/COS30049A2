"""The training-stage gate (explorer/stages.py; docs/TRAINING_PLAN.md section 3)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from explorer import stages


def test_paired_difference_sign_and_interval():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 10)
    before = rng.random(200) < 0.5
    after = before | (rng.random(200) < 0.3)  # strictly better on some units, never worse
    diff, low, high = stages.paired_difference(before, after, groups)
    assert diff == pytest.approx(after.mean() - before.mean())
    assert 0 < low <= diff <= high


def test_paired_difference_identical_is_zero():
    x = np.array([1, 0, 1, 1, 0, 0], dtype=float)
    diff, low, high = stages.paired_difference(x, x, np.array([0, 0, 1, 1, 2, 2]))
    assert diff == low == high == 0


def test_paired_difference_needs_aligned_units():
    with pytest.raises(ValueError):
        stages.paired_difference(np.zeros(3), np.zeros(4), np.zeros(3))


def test_mcnemar_counts_only_discordant_pairs():
    before = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    after = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    assert stages.mcnemar_p(before, after) == pytest.approx(2 * 0.5 ** 7)  # 7 discordant, all one way
    assert stages.mcnemar_p(before, before) == 1.0


def test_decide_lists_every_failed_rule():
    ok = stages.decide(0.01, {"spike": 0.8}, {"spike": 0.79}, {"spike": 0.02}, loss_agrees=True,
                       guardrail_before=0.9, guardrail_after=0.9, flagged_share=0.1)
    assert ok.accepted and not ok.reasons
    bad = stages.decide(-0.01, {"spike": 0.8}, {"spike": 0.7}, {"spike": 0.02}, loss_agrees=False,
                        guardrail_before=0.9, guardrail_after=0.8, flagged_share=0.5,
                        address_share_before=0.45, address_share_after=0.5)
    assert not bad.accepted and len(bad.reasons) == 6
    already_over = stages.decide(0.01, {}, {}, {}, loss_agrees=True, address_share_before=0.45,
                                 address_share_after=0.45)
    assert already_over.accepted  # an address over the limit at M0 does not block a stage that leaves it there


def test_freeze_then_check(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        check_freeze(Path(tmp), monkeypatch)


def check_freeze(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "test.jsonl"
    f.write_text("a\n", encoding="utf-8")
    monkeypatch.setattr(stages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(stages.SETTINGS, "stages", {**stages.SETTINGS["stages"], "frozenFiles": ["*.jsonl"]})
    manifest = tmp_path / "frozen.json"
    assert list(stages.freeze(manifest)) == ["test.jsonl"]
    with pytest.raises(FileExistsError):
        stages.freeze(manifest)
    stages.check_frozen(manifest)
    f.write_text("b\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        stages.check_frozen(manifest)


def test_record_and_last_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        check_record(Path(tmp) / "stages.csv")


def check_record(table: Path) -> None:
    stages.record({"family": "scoring", "model": "baseline", "stage": "M0", "accepted": True, "commit": "abc"}, table)
    stages.record({"family": "scoring", "model": "baseline", "stage": "C1", "accepted": False, "commit": "def"}, table)
    assert len(pd.read_csv(table)) == 2
    assert stages.last_accepted("scoring", "baseline", table)["stage"] == "M0"
    with pytest.raises(KeyError):
        stages.record({"not_a_column": 1}, table)


def test_frozen_files_exist():
    assert stages.frozen_paths()  # every pattern in settings.json matches a file in the repository


def test_compare_accepts_a_clear_gain_and_records_it(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        check_compare(Path(tmp), monkeypatch)


def check_compare(tmp: Path, monkeypatch) -> None:
    rng = np.random.default_rng(1)
    rows = [{"method": "baseline", "seed": 40000, "pattern": p, "plant_id": f"{p}{i}", "address": f"a{i % 20}"}
            for p in ("spike", "burst", "dormant") for i in range(60)]
    before = pd.DataFrame(rows).assign(max=rng.random(180) < 0.5)
    before["mean"] = before["max"].astype(float)
    after = before.copy()
    after.loc[after["pattern"] == "burst", ["max", "mean"]] = [True, 1.0]  # burst fixed, nothing else moves
    monkeypatch.setattr(stages, "DEV_DIR", tmp)
    for name, hits in (("M0", before), ("A0", after)):
        hits.to_csv(tmp / f"{name}_hits.csv", index=False)
        recalls = {p: hits.loc[hits["pattern"] == p, "max"].mean() for p in ("spike", "burst", "dormant")}
        pd.DataFrame([{"method": "baseline", **{f"recall_{p}": r for p, r in recalls.items()},
                       "mean_recall": np.mean(list(recalls.values())), "flag_rate": 0.02,
                       "max_address_share": 0.45}]).to_csv(
            tmp / f"{name}_summary.csv", index=False)
    table = tmp / "stages.csv"
    monkeypatch.setitem(stages.SETTINGS, "stages", {**stages.SETTINGS["stages"], "table": str(table)})
    stages.run_compare("M0", "A0", "A0")
    row = pd.read_csv(table).iloc[-1]
    assert row["stage"] == "A0" and bool(row["accepted"]) and row["diff_previous_ci_low"] > 0
