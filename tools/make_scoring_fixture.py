"""Build tests/fixtures/scoring_parity.json: seeded histories scored by the A1 prototype itself.

tests/test_features.py checks that explorer/features.py (the Python port) gives the same features,
windows, z-scores and flags as reference/a1_prototype/scoring.ts on these histories, which cover
the cases where a port is easiest to get wrong: same-block ties, a constant history (zero spread),
mixed directions, a dormant address waking up, a burst, a spike and a short history.

Needs Node.js 22.6 or later (it runs the TypeScript directly). Only needed to rebuild the fixture:
    python tools/make_scoring_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORING_TS = ROOT / "reference" / "a1_prototype" / "scoring.ts"
SETTINGS_JSON = ROOT / "config" / "settings.json"
FIXTURE = ROOT / "tests" / "fixtures" / "scoring_parity.json"
SEED = 30049
DAY, HOUR = 86_400, 3600
T0 = 1_760_000_000  # 2025-10-09 UTC; any fixed time works

RUNNER = """
import { readFileSync } from 'node:fs';
import { scoreAddress } from './scoring.ts';
const input = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = {};
for (const [address, txs] of Object.entries(input)) out[address] = scoreAddress(txs);
process.stdout.write(JSON.stringify(out));
"""


def txid(address: str, i: int) -> str:
    # Lowercase hex: JavaScript's localeCompare and Python's string order agree on it.
    return hashlib.sha256(f"{address}/{i}".encode()).hexdigest()


def histories(rng: random.Random) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    def add(name: str, events: list[tuple[int, int, int]]) -> None:
        out[name] = [{"txid": txid(name, i), "time": t, "blockHeight": 900_000 + (t - T0) // 600,
                      "receivedSats": r, "sentSats": s} for i, (t, r, s) in enumerate(events)]

    # Collects weekly, sweeps monthly: the case CR-04's same-direction window exists for.
    ev, bal = [], 0
    for d in range(0, 200):
        if d % 7 == 0:
            amt = int(rng.lognormvariate(math.log(50_000), 0.4))
            ev.append((T0 + d * DAY + rng.randrange(HOUR), amt, 0))
            bal += amt
        if d % 30 == 29 and bal:
            ev.append((T0 + d * DAY + 2 * HOUR, 0, bal))
            bal = 0
    add("collector", ev)

    # Ordinary amounts with one value spike.
    ev = [(T0 + d * DAY + rng.randrange(DAY), int(rng.lognormvariate(math.log(200_000), 0.6)), 0) for d in range(0, 120, 2)]
    ev.append((T0 + 101 * DAY, 60_000_000, 0))
    add("spike", ev)

    # Regular activity with ten transactions inside one day.
    ev = [(T0 + d * DAY, 0, int(rng.lognormvariate(math.log(80_000), 0.5))) for d in range(0, 150, 3)]
    ev += [(T0 + 120 * DAY + k * 2 * HOUR, 0, int(rng.lognormvariate(math.log(80_000), 0.5))) for k in range(10)]
    add("burst", ev)

    # Active, then silent for 100 days, then one more transaction.
    ev = [(T0 + d * DAY + rng.randrange(DAY), int(rng.lognormvariate(math.log(30_000), 0.3)), 0) for d in range(60)]
    ev.append((T0 + 160 * DAY, 31_000, 0))
    add("dormant", ev)

    # Same-block ties, identical amounts (zero spread, so the floor applies) and a no-net transaction.
    ev = []
    for d in range(0, 90, 3):
        t = T0 + d * DAY
        ev += [(t, 10_000, 0), (t, 10_000, 0)]
        if d % 9 == 0:
            ev.append((t, 0, 20_000))
    ev += [(T0 + 91 * DAY, 5_000, 5_000), (T0 + 92 * DAY, 400_000, 0)]
    add("ties_constant", ev)

    # Too short to score.
    add("short", [(T0 + d * DAY, 1_000 * (d + 1), 0) for d in range(5)])
    return out



def main() -> int:
    rng = random.Random(SEED)
    inputs = histories(rng)
    settings = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        (tmp_dir / "settings.ts").write_text(
            "import raw from './settings.json' with { type: 'json' };\n"
            "export const settings = raw;\n"
            "export type FeatureId = 'F1' | 'F2' | 'F3' | 'F4';\n"
            "export const FEATURE_IDS: FeatureId[] = ['F1', 'F2', 'F3', 'F4'];\n",
            encoding="utf-8",
        )
        source = SCORING_TS.read_text(encoding="utf-8")
        source = re.sub(r"from '\.\./\.\./lib/settings';", "from './settings.ts';", source)
        (tmp_dir / "scoring.ts").write_text(source, encoding="utf-8")
        (tmp_dir / "run.ts").write_text(RUNNER, encoding="utf-8")
        (tmp_dir / "input.json").write_text(json.dumps(inputs), encoding="utf-8")
        result = subprocess.run(["node", "--no-warnings", str(tmp_dir / "run.ts"), str(tmp_dir / "input.json")],
                                capture_output=True, text=True, cwd=tmp_dir)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    fixture = {
        "about": "Inputs and the A1 prototype's scoreAddress output (reference/a1_prototype/scoring.ts), "
                 "built by tools/make_scoring_fixture.py.",
        "settings": {"features": settings["features"], "baseline": settings["baseline"]},
        "inputs": inputs,
        "expected": json.loads(result.stdout),
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    n = sum(len(v) for v in inputs.values())
    flagged = sum(t["flagged"] for v in fixture["expected"].values() for t in v)
    print(f"Wrote {FIXTURE.relative_to(ROOT)}: {len(inputs)} histories, {n} transactions, {flagged} flagged by the prototype")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
