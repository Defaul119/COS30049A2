// Illustrative data for the A1 prototype. Every address, transaction and amount here is
// invented: addresses are random but correctly checksummed (so the validator accepts them),
// and each history is simulated with a seeded random generator so every run is identical.
// A few patterns the method should catch are planted on purpose: a value spike, a burst,
// a dormant address waking up, a dust-sized payment, and two short histories.

import { base58CheckEncode, segwitEncode, type AddressKind } from '../../lib/address';
import type { RawTx } from './scoring';

export const SNAPSHOT_DATE = '2026-10-05';
const END = Date.parse(`${SNAPSHOT_DATE}T00:00:00Z`) / 1000;
const DAY = 86_400;
const HOUR = 3600;
const SATS = 100_000_000;
// Block heights are estimated from time around a real reference block (~10 minutes per block).
const REF_TIME = Date.parse('2026-10-07T01:16:27Z') / 1000;
const REF_HEIGHT = 970_267;

export interface IllustrativeAddress {
  address: string;
  kind: AddressKind;
  profile: string;
  txs: RawTx[];
}

type Rng = () => number;

function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

const lognormal = (rng: Rng, med: number, sigma: number) => med * Math.exp(sigma * normal(rng));

function randomBytes(rng: Rng, n: number): Uint8Array {
  return Uint8Array.from({ length: n }, () => Math.floor(rng() * 256));
}

const hex = (bytes: Uint8Array) => [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');

function makeAddress(rng: Rng, kind: AddressKind): string {
  switch (kind) {
    case 'p2pkh':
      return base58CheckEncode(0x00, randomBytes(rng, 20));
    case 'p2sh':
      return base58CheckEncode(0x05, randomBytes(rng, 20));
    case 'p2tr':
      return segwitEncode(1, randomBytes(rng, 32));
    default:
      return segwitEncode(0, randomBytes(rng, 20));
  }
}

/** Keeps a running balance so outgoing amounts are plausible: an "out" spends the whole balance, sometimes with change back. */
class Wallet {
  balance = 0;
  readonly txs: RawTx[] = [];
  constructor(private rng: Rng) {}

  receive(time: number, sats: number) {
    this.push(time, Math.max(546, Math.round(sats)), 0);
    this.balance += Math.max(546, Math.round(sats));
  }

  spend(time: number, share = 1) {
    if (this.balance < 10_000) return;
    const payment = Math.round(this.balance * share) - 1_500;
    const change = this.balance - payment - 1_500;
    const keepChange = change > 0 && (share < 1 || this.rng() < 0.3);
    this.push(time, keepChange ? change : 0, this.balance);
    this.balance = keepChange ? change : 0;
  }

  private push(time: number, receivedSats: number, sentSats: number) {
    const t = Math.round(time);
    this.txs.push({
      txid: hex(randomBytes(this.rng, 32)),
      time: t,
      blockHeight: REF_HEIGHT + Math.round((t - REF_TIME) / 600),
      receivedSats,
      sentSats,
    });
  }
}

/** Times from start to end with log-normally distributed gaps (median gapHours). */
function activityTimes(rng: Rng, start: number, end: number, gapHours: number, sigma: number): number[] {
  const out: number[] = [];
  for (let t = start; t < end; t += lognormal(rng, gapHours, sigma) * HOUR) out.push(t);
  return out;
}

function burstTimes(rng: Rng, dayStart: number, count: number, hours: number): number[] {
  return Array.from({ length: count }, () => dayStart + 2 * HOUR + rng() * hours * HOUR).sort((a, b) => a - b);
}

interface Profile {
  profile: string;
  kind: AddressKind;
  seed: number;
  build: (w: Wallet, rng: Rng) => void;
}

const PROFILES: Profile[] = [
  {
    profile: 'Regular small payments in, periodic sweeps out; one large payment and one burst planted',
    kind: 'p2wpkh',
    seed: 101,
    build(w, rng) {
      const start = END - 178 * DAY;
      const spike = END - 52 * DAY + 9 * HOUR;
      const burstDay = END - 24 * DAY;
      let n = 0;
      let planted = false;
      for (const t of activityTimes(rng, start, END - DAY, 30, 0.55)) {
        if (!planted && t > spike) {
          w.receive(spike, 4_400_000);
          planted = true;
        }
        if (Math.abs(t - burstDay) < DAY) continue;
        if (++n % 12 === 0) w.spend(t);
        else w.receive(t, lognormal(rng, 180_000, 0.35));
      }
      for (const t of burstTimes(rng, burstDay, 11, 9)) w.receive(t, lognormal(rng, 160_000, 0.3));
    },
  },
  {
    profile: 'Steady activity, then 80 days without any transaction, then wakes up',
    kind: 'p2pkh',
    seed: 202,
    build(w, rng) {
      const start = END - 205 * DAY;
      const sleep = END - 98 * DAY;
      let n = 0;
      for (const t of activityTimes(rng, start, sleep, 36, 0.5)) {
        if (++n % 8 === 0) w.spend(t, 0.4);
        else w.receive(t, lognormal(rng, 1_000_000, 0.45));
      }
      const wake = END - 18 * DAY + 14 * HOUR;
      w.spend(wake);
      for (const t of activityTimes(rng, wake + 30 * HOUR, END - DAY, 40, 0.5)) w.receive(t, lognormal(rng, 1_000_000, 0.45));
    },
  },
  {
    profile: 'Busy address: several transactions a day, in and out; one very large payment out planted',
    kind: 'p2wpkh',
    seed: 303,
    build(w, rng) {
      const start = END - 122 * DAY;
      const big = END - 41 * DAY + 15 * HOUR;
      let planted = false;
      for (const t of activityTimes(rng, start, END - DAY, 6, 0.75)) {
        if (!planted && t > big) {
          w.receive(big - 3 * HOUR, 140 * SATS);
          w.spend(big);
          planted = true;
        }
        if (rng() < 0.42) w.spend(t, 0.2 + rng() * 0.15);
        else w.receive(t, lognormal(rng, 5_000_000, 0.6));
      }
    },
  },
  {
    profile: 'Occasional use every few days; one dust-sized payment planted',
    kind: 'p2sh',
    seed: 404,
    build(w, rng) {
      const start = END - 262 * DAY;
      const dust = END - 70 * DAY + 4 * HOUR;
      let done = false;
      let n = 0;
      for (const t of activityTimes(rng, start, END - 2 * DAY, 96, 0.5)) {
        if (!done && t > dust) {
          w.receive(dust, 546);
          done = true;
        }
        if (++n % 5 === 0) w.spend(t, 0.6);
        else w.receive(t, lognormal(rng, 2_500_000, 0.4));
      }
    },
  },
  {
    profile: 'Weekly payments of a similar size for a year; one short burst planted',
    kind: 'p2tr',
    seed: 505,
    build(w, rng) {
      const start = END - 392 * DAY;
      const burstDay = END - 63 * DAY;
      let n = 0;
      for (const t of activityTimes(rng, start, END - 3 * DAY, 168, 0.12)) {
        if (Math.abs(t - burstDay) < 3 * DAY) continue;
        if (++n % 9 === 0) w.spend(t, 0.5);
        else w.receive(t, lognormal(rng, 25_000_000, 0.1));
      }
      for (const t of burstTimes(rng, burstDay, 5, 40)) w.receive(t, lognormal(rng, 25_000_000, 0.1));
    },
  },
  {
    profile: 'Short history: 6 transactions over 9 days (too little to score)',
    kind: 'p2wpkh',
    seed: 606,
    build(w, rng) {
      const start = END - 12 * DAY;
      for (let i = 0; i < 6; i++) {
        const t = start + i * 1.8 * DAY + rng() * 6 * HOUR;
        if (i === 4) w.spend(t, 0.7);
        else w.receive(t, lognormal(rng, 800_000, 0.4));
      }
    },
  },
  {
    profile: 'Short history: 4 transactions over 3 days (too little to score)',
    kind: 'p2pkh',
    seed: 707,
    build(w, rng) {
      const start = END - 6 * DAY;
      for (let i = 0; i < 4; i++) w.receive(start + i * 20 * HOUR + rng() * 3 * HOUR, lognormal(rng, 300_000, 0.3));
    },
  },
];

export function generateIllustrativeData(): IllustrativeAddress[] {
  return PROFILES.map((p) => {
    const rng = mulberry32(p.seed);
    const address = makeAddress(rng, p.kind);
    const wallet = new Wallet(rng);
    p.build(wallet, rng);
    return { address, kind: p.kind, profile: p.profile, txs: wallet.txs };
  });
}
