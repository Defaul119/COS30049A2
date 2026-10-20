// Every threshold lives in config/settings.json, shared with the Python pipeline.
import raw from '../../../config/settings.json';

export const settings = raw;

export const LIMITATION_LINE: string = raw.limitationLine;

export type FeatureId = 'F1' | 'F2' | 'F3' | 'F4';
export const FEATURE_IDS: FeatureId[] = ['F1', 'F2', 'F3', 'F4'];
