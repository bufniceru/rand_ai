import type { PredictionAuditRecord, StrategyEfficacyRecord, StrategyId } from "../types";

import { strategyNames } from "./strategyNames";

export type HitHistoryScope = "all" | 100 | 250 | 500;
export type RelationshipMeasure = "sharedNumbers" | "bothHitDraws";
export interface HitDraw {
  record: PredictionAuditRecord;
  matches: Partial<Record<StrategyId, number[]>>;
}
export interface HitPair {
  left: StrategyId;
  right: StrategyId;
  sharedNumbers: number;
  sharedDraws: number;
  bothHitDraws: number;
}

export function strategyHitStatistics(
  audit: PredictionAuditRecord[], efficacy: StrategyEfficacyRecord[],
  enabled: StrategyId[], scope: HitHistoryScope = "all",
) {
  const names = new Map<StrategyId, string>(enabled.map(id => [id, strategyNames[id]]));
  for (const record of efficacy) {
    for (const id of Object.keys(record.strategyHits) as StrategyId[]) names.set(id, names.get(id) ?? strategyNames[id]);
  }
  for (const record of audit) for (const item of record.numbers) {
    for (const strategy of item.strategies) names.set(strategy.id, strategy.name.replace(/ \(Experimental\)$/, ""));
  }
  const ids = [...names.keys()];
  const records = scope === "all" ? audit : audit.slice(-scope);
  const draws: HitDraw[] = records.map(record => {
    const matches: HitDraw["matches"] = {};
    for (const id of ids) matches[id] = [];
    for (const item of record.numbers) for (const strategy of item.strategies) {
      const numbers = matches[strategy.id]!;
      if (!numbers.includes(item.number)) numbers.push(item.number);
    }
    return { record, matches };
  });
  const evaluated = new Map(efficacy.map(record => [record.targetDrawNumber, record]));
  const strategies = ids.map(id => {
    // Missing strategy records are not zero-hit evaluations.
    const counts = draws.filter(draw => Object.hasOwn(evaluated.get(draw.record.targetDrawNumber)?.strategyHits ?? {}, id))
      .map(draw => draw.matches[id]!.length);
    const hits = counts.reduce((total, count) => total + count, 0);
    return { id, name: names.get(id)!, hits, evaluatedDraws: counts.length,
      hitDraws: counts.filter(count => count > 0).length,
      average: counts.length ? hits / counts.length : 0,
      buckets: Array.from({ length: 7 }, (_, count) => counts.filter(value => value === count).length) };
  });
  const pairs: HitPair[] = [];
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const left = ids[i], right = ids[j];
    let sharedNumbers = 0, sharedDraws = 0, bothHitDraws = 0;
    for (const draw of draws) {
      const a = draw.matches[left]!, b = draw.matches[right]!;
      const shared = a.filter(number => b.includes(number)).length;
      sharedNumbers += shared;
      if (shared) sharedDraws++;
      if (a.length && b.length) bothHitDraws++;
    }
    pairs.push({ left, right, sharedNumbers, sharedDraws, bothHitDraws });
  }
  return { draws, strategies, pairs, names };
}

export function contributingDraws(draws: HitDraw[], pair: HitPair, measure: RelationshipMeasure) {
  return draws.filter(draw => {
    const a = draw.matches[pair.left] ?? [], b = draw.matches[pair.right] ?? [];
    return measure === "sharedNumbers" ? a.some(number => b.includes(number)) : a.length > 0 && b.length > 0;
  });
}
