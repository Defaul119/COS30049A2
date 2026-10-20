// The API contract (report §2.6). The prototype's mock and the future FastAPI back end
// both return exactly these shapes, so switching from one to the other changes no screen code.
//
//   GET  /api/addresses                                  -> AddressesResponse
//   POST /api/lookup            {address}                -> LookupResponse | 400 InvalidBody | 404 UnsupportedBody
//   GET  /api/addresses/{address}/timeline               -> TimelineResponse | 404
//   GET  /api/addresses/{address}/transactions?date=     -> DayResponse | 400 | 404
//   GET  /api/transactions/{txid}?address=               -> TransactionDetail | 404 | 422 InsufficientHistoryBody
//
// Times are ISO 8601 in UTC; days are 'YYYY-MM-DD' in UTC; amounts are integer satoshis.

import type { FeatureId } from '../lib/settings';

export type Direction = 'in' | 'out' | 'none';

export interface SnapshotInfo {
  snapshotDate: string;
  source: string;
  sourceUrl: string;
  illustrative: boolean;
  dayTimeZone: 'UTC';
  addressCount: number;
  method: MethodInfo;
}

export interface MethodInfo {
  id: string;
  name: string;
  flagAbsZ: number;
  contributingAbsZ: number;
  note: string;
}

export interface HistoryRequirement {
  minTx: number;
  minSpanDays: number;
}

export interface AddressSummary {
  address: string;
  firstSeen: string;
  lastSeen: string;
  txCount: number;
  scoredCount: number;
  flaggedCount: number;
  history: 'sufficient' | 'insufficient';
}

export interface AddressesResponse {
  snapshot: SnapshotInfo;
  addresses: AddressSummary[];
}

export interface LookupResponse {
  status: 'supported';
  address: string;
  summary: AddressSummary;
}

export interface TimelineDay {
  date: string;
  txCount: number;
  flaggedCount: number;
}

export interface TimelineResponse {
  address: string;
  snapshot: SnapshotInfo;
  range: { start: string; end: string };
  /** Only days with at least one transaction, oldest first. */
  days: TimelineDay[];
  totals: { txCount: number; scoredCount: number; flaggedCount: number; flaggedDays: number };
  history: {
    status: 'sufficient' | 'insufficient';
    txCount: number;
    spanDays: number;
    required: HistoryRequirement;
    firstScoredDate: string | null;
  };
}

export type ScoreStatus = 'flagged' | 'not-flagged' | 'not-scored';

export interface TransactionRow {
  txid: string;
  time: string;
  direction: Direction;
  netSats: number;
  status: ScoreStatus;
}

export interface DayResponse {
  address: string;
  date: string;
  transactions: TransactionRow[];
}

/** The facts every transaction has, scored or not (FR-03). */
export interface TransactionCore {
  txid: string;
  address: string;
  time: string;
  blockHeight: number;
  direction: Direction;
  receivedSats: number;
  sentSats: number;
  netSats: number;
  /** Public block-explorer page; null for illustrative data, which has no public record. */
  sourceUrl: string | null;
}

export type FeatureRole = 'flag' | 'contributing' | 'within-usual' | 'no-comparison';

export interface FeatureDetail {
  id: FeatureId;
  name: string;
  /** What the feature measures, in plain words. */
  description: string;
  /** Value in human units: BTC for F1, count for F2, hours for F3, ratio for F4. */
  value: number | null;
  /** Median of the comparison window, same units as value. */
  usual: number | null;
  z: number | null;
  role: FeatureRole;
}

/** What a group of features was compared against (AC-03 "comparison basis"). */
export interface ComparisonBasis {
  features: FeatureId[];
  label: string;
  description: string;
  txCount: number;
  start: string;
  end: string;
}

export interface TransactionDetail extends TransactionCore {
  scored: true;
  flagged: boolean;
  score: number;
  method: MethodInfo;
  features: FeatureDetail[];
  comparison: ComparisonBasis[];
  limitation: string;
}

// ------------------------------------------------------------------ error bodies

export interface InvalidBody {
  status: 'invalid';
  message: string;
  hint: string;
}

export interface UnsupportedBody {
  status: 'unsupported';
  message: string;
  examples: string[];
}

export interface InsufficientHistoryBody {
  status: 'insufficient_history';
  message: string;
  transaction: TransactionCore;
  history: { earlierTx: number; spanDays: number; required: HistoryRequirement };
  firstScoredDate: string | null;
}

export interface NotFoundBody {
  status: 'not_found';
  message: string;
}

export interface NetworkBody {
  status: 'network';
  message: string;
}

export type ApiErrorBody = InvalidBody | UnsupportedBody | InsufficientHistoryBody | NotFoundBody | NetworkBody;

/** Thrown by every API call that does not succeed. status 0 means the request never got an answer. */
export class ApiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

export interface ExplorerApi {
  listAddresses(): Promise<AddressesResponse>;
  lookup(address: string): Promise<LookupResponse>;
  timeline(address: string): Promise<TimelineResponse>;
  day(address: string, date: string): Promise<DayResponse>;
  transaction(txid: string, address: string): Promise<TransactionDetail>;
}
