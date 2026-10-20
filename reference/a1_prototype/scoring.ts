// Reference implementation of the features and the baseline (report §2.3, demo-plan Phases 2 and 4).
// The prototype uses it to score its illustrative data; the Python pipeline implements the
// same rules for the real snapshot, and both read their thresholds from config/settings.json.
//
// Time safety (DR-02): everything computed for a transaction uses only transactions
// confirmed STRICTLY EARLIER. Two transactions in the same block share a time and are
// never part of each other's history.
//
// Two comparison windows, both "the previous 30 days, extended back to at least the last
// 20 transactions" (CR-02):
//   timing window - all earlier transactions; used for F2 Burst and F3 Gap
//   size window   - earlier transactions in the same direction (in vs out) when
//                   settings.features.sizeComparison is "same-direction" (proposed CR-04),
//                   otherwise the timing window; used for F1 Size and F4 Relative size.
// Without CR-04, an address that collects small payments and sweeps them out every few
// weeks has every sweep flagged, because each sweep is compared with small incoming amounts.

import { settings, FEATURE_IDS, type FeatureId } from '../../lib/settings';

const F = settings.features;
const B = settings.baseline;
const HOUR = 3600;
const DAY = 86_400;
const SIZE_FEATURES: FeatureId[] = ['F1', 'F4'];

export interface RawTx {
  txid: string;
  time: number; // unix seconds, UTC
  blockHeight: number;
  receivedSats: number;
  sentSats: number;
}

export interface ComparisonWindow {
  /** Transactions in the window. */
  count: number;
  /** How many of them fall inside the 30-day period itself. */
  countInWindowDays: number;
  /** True when the window had to reach back past 30 days to collect enough transactions. */
  extended: boolean;
  /** 'all', or the direction the size window was restricted to. */
  direction: 'all' | 'in' | 'out' | 'none';
  start: number;
  end: number;
}

export interface ScoredTx extends RawTx {
  netSats: number;
  earlierCount: number;
  historySpanDays: number;
  scored: boolean;
  /** Feature values: F1 log10(1+|net| sats), F2 count, F3 hours, F4 ratio. */
  features: Record<FeatureId, number | null>;
  /** Median of the comparison window in readable units: F1 sats, F2 count, F3 hours, F4 ratio. */
  usual: Record<FeatureId, number | null>;
  z: Record<FeatureId, number | null>;
  timingWindow: ComparisonWindow | null;
  sizeWindow: ComparisonWindow | null;
  flagged: boolean;
  score: number;
}

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Robust z of `value` against `sample`, with a floor on the scale so a constant history can't divide by zero. */
export function robustZ(value: number, sample: number[], minScale: number): { z: number; center: number } {
  const center = median(sample);
  const mad = median(sample.map((v) => Math.abs(v - center)));
  const scale = Math.max(B.madScale * mad, minScale);
  return { z: (value - center) / scale, center };
}

/** The scale on which the baseline compares each feature (heavy-tailed ones are logged). */
function transform(id: FeatureId, v: number | null): number | null {
  if (v === null) return null;
  if (id === 'F3') return Math.log10(1 + v);
  if (id === 'F4') return Math.log10(Math.max(v, 0.001));
  return v;
}

/** Back from the comparison scale to readable units, for the "usual" column. */
function readable(id: FeatureId, t: number): number {
  if (id === 'F1' || id === 'F3') return 10 ** t - 1; // satoshis, hours
  if (id === 'F4') return 10 ** t; // ratio
  return t;
}

type Dir = 'in' | 'out' | 'none';
const direction = (t: RawTx): Dir => (t.receivedSats > t.sentSats ? 'in' : t.receivedSats < t.sentSats ? 'out' : 'none');

/**
 * Indices of earlier transactions (newest first) that pass `keep`: every one in the last
 * comparisonWindowDays, then older ones until at least comparisonMinTx are collected.
 */
function collectWindow(txs: RawTx[], earlierEnd: number, now: number, keep: (j: number) => boolean) {
  const idx: number[] = [];
  let inPeriod = 0;
  for (let j = earlierEnd - 1; j >= 0; j--) {
    if (!keep(j)) continue;
    const recent = txs[j].time >= now - F.comparisonWindowDays * DAY;
    if (!recent && idx.length >= F.comparisonMinTx) break;
    if (recent) inPeriod++;
    idx.push(j);
  }
  return { idx, inPeriod };
}

function describe(txs: RawTx[], w: { idx: number[]; inPeriod: number }, dir: ComparisonWindow['direction']): ComparisonWindow | null {
  if (w.idx.length === 0) return null;
  return {
    count: w.idx.length,
    countInWindowDays: w.inPeriod,
    extended: w.idx.length > w.inPeriod,
    direction: dir,
    start: txs[w.idx[w.idx.length - 1]].time,
    end: txs[w.idx[0]].time,
  };
}

export function scoreAddress(input: RawTx[]): ScoredTx[] {
  const txs = [...input].sort((a, b) => a.time - b.time || a.txid.localeCompare(b.txid));
  const absNet = txs.map((t) => Math.abs(t.receivedSats - t.sentSats));
  const dirs = txs.map(direction);
  const sameDirection = F.sizeComparison === 'same-direction';

  // earlier transactions of i are indices [0, earlierEnd[i]): same-time transactions are excluded.
  const earlierEnd: number[] = [];
  txs.forEach((t, i) => {
    earlierEnd[i] = i > 0 && txs[i - 1].time === t.time ? earlierEnd[i - 1] : i;
  });

  // Pass 1: raw features and the two comparison windows.
  const timing = txs.map((t, i) => collectWindow(txs, earlierEnd[i], t.time, () => true));
  const size = txs.map((t, i) =>
    sameDirection ? collectWindow(txs, earlierEnd[i], t.time, (j) => dirs[j] === dirs[i]) : timing[i],
  );
  const features = txs.map((t, i) => {
    const e = earlierEnd[i];
    let burst = 0;
    for (let j = e - 1; j >= 0 && txs[j].time >= t.time - F.burstWindowDays * DAY; j--) burst++;
    const usualAbs = size[i].idx.length ? median(size[i].idx.map((j) => absNet[j])) : null;
    return {
      F1: Math.log10(1 + absNet[i]),
      F2: burst,
      F3: e > 0 ? (t.time - txs[e - 1].time) / HOUR : null,
      F4: usualAbs === null ? null : absNet[i] / Math.max(usualAbs, 1),
    } as Record<FeatureId, number | null>;
  });

  // Pass 2: robust z of each feature against the same feature over its comparison window.
  return txs.map((t, i) => {
    const e = earlierEnd[i];
    const historySpanDays = e > 0 ? (t.time - txs[0].time) / DAY : 0;
    const scored = e >= F.minHistoryTx && historySpanDays >= F.minHistorySpanDays;
    const z = {} as Record<FeatureId, number | null>;
    const usual = {} as Record<FeatureId, number | null>;

    for (const id of FEATURE_IDS) {
      const window = SIZE_FEATURES.includes(id) ? size[i] : timing[i];
      const own = transform(id, features[i][id]);
      const sample = window.idx.map((j) => transform(id, features[j][id])).filter((v): v is number => v !== null);
      usual[id] = sample.length ? readable(id, median(sample)) : null;
      z[id] = own === null || sample.length < F.minComparisonValues ? null : robustZ(own, sample, B.minScale[id]).z;
    }

    const zs = Object.values(z).filter((v): v is number => v !== null);
    const score = scored && zs.length ? Math.max(...zs.map(Math.abs)) : 0;
    return {
      ...t,
      netSats: t.receivedSats - t.sentSats,
      earlierCount: e,
      historySpanDays,
      scored,
      features: features[i],
      usual,
      z,
      timingWindow: describe(txs, timing[i], 'all'),
      sizeWindow: describe(txs, size[i], sameDirection ? dirs[i] : 'all'),
      flagged: scored && score >= B.flagAbsZ,
      score,
    };
  });
}
