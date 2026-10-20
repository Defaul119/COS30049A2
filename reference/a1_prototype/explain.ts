// Turns feature values into the plain-language reasons shown in the detail panel (FR-04, AC-03).
// Wording rule (NFR-01): describe what is unusual, never what it might mean about a person.

import type { FeatureDetail } from '../api/types';
import { formatBtcShort, formatHours, formatRatio, formatZ, plural } from './format';

export interface Reason {
  id: FeatureDetail['id'];
  headline: string;
  detail: string;
  z: number;
  level: 'flag' | 'contributing' | 'largest';
}

const SATS = 100_000_000;

function headline(f: FeatureDetail, z: number): string {
  const up = z > 0;
  switch (f.id) {
    case 'F1':
      return up ? 'Larger amount than usual' : 'Smaller amount than usual';
    case 'F2':
      return up ? 'Busier week than usual' : 'Quieter week than usual';
    case 'F3':
      return up ? 'Longer gap than usual' : 'Shorter gap than usual';
    case 'F4':
      return up ? 'Large compared with the usual amount' : 'Small compared with the usual amount';
  }
}

export function describeValue(f: FeatureDetail, which: 'value' | 'usual'): string {
  const v = which === 'value' ? f.value : f.usual;
  if (v === null) return '—';
  switch (f.id) {
    case 'F1':
      return formatBtcShort(v * SATS);
    case 'F2':
      return which === 'usual' ? `${Number(v.toFixed(1))} transactions` : plural(v, 'transaction');
    case 'F3':
      return formatHours(v);
    case 'F4':
      return formatRatio(v);
  }
}

function detail(f: FeatureDetail): string {
  const value = describeValue(f, 'value');
  const usual = describeValue(f, 'usual');
  switch (f.id) {
    case 'F1':
      return `${value} moved, against a usual ${usual}.`;
    case 'F2':
      return `${value} in the previous 7 days, against a usual ${usual}.`;
    case 'F3':
      return `${value} since the previous transaction, against a usual ${usual}.`;
    case 'F4':
      return `${value} the usual amount for this direction.`;
  }
}

/**
 * Features at or above the contributing level, strongest first. If an ML method flags a
 * transaction where no single feature reaches that level, fall back to the two largest
 * deviations, so a flagged transaction always has an explanation (AC-03).
 */
export function reasonsFor(features: FeatureDetail[]): { reasons: Reason[]; fallback: boolean } {
  const scored = features.filter((f): f is FeatureDetail & { z: number } => f.z !== null);
  const strong = scored.filter((f) => f.role === 'flag' || f.role === 'contributing');
  const fallback = strong.length === 0;
  const chosen = (fallback ? scored : strong).sort((a, b) => Math.abs(b.z) - Math.abs(a.z)).slice(0, fallback ? 2 : 4);
  return {
    fallback,
    reasons: chosen.map((f) => ({
      id: f.id,
      headline: headline(f, f.z),
      detail: detail(f),
      z: f.z,
      level: fallback ? 'largest' : (f.role as 'flag' | 'contributing'),
    })),
  };
}

export const LEVEL_TEXT: Record<Reason['level'], string> = {
  flag: 'reached the flag level',
  contributing: 'contributing',
  largest: 'largest difference',
};

export { formatZ };
