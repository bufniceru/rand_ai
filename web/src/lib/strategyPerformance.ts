import type { StrategyEfficacyRecord, StrategyId } from "../types";

export function strategyPerformance(records: StrategyEfficacyRecord[], id: StrategyId) {
  const hits = records.map(record => record.strategyHits[id]).filter((value): value is number => value !== undefined);
  return { drawCount: hits.length, hits: hits.reduce((sum, value) => sum + value, 0),
    average: hits.length ? hits.reduce((sum, value) => sum + value, 0) / hits.length : 0,
    distribution: Array.from({ length: 7 }, (_, count) => hits.filter(value => value === count).length) };
}

export function strategyHitCurve(records: StrategyEfficacyRecord[], id: StrategyId, window?: number): (number | null)[] {
  if (window === undefined) {
    let total = 0;
    let count = 0;
    return records.map(record => {
      const hits = record.strategyHits[id];
      if (hits === undefined) return null;
      total += hits;
      count += 1;
      return total / count;
    });
  }
  return records.map((record, index) => {
    if (record.strategyHits[id] === undefined) return null;
    const start = Math.max(0, index - window + 1);
    const hits = records.slice(start, index + 1).map(item => item.strategyHits[id])
      .filter((value): value is number => value !== undefined);
    return hits.reduce((sum, value) => sum + value, 0) / hits.length;
  });
}
