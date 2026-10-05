import { describe, expect, it } from "vitest";
import { contributingDraws, strategyHitStatistics } from "./strategyHitStatistics";
import type { PredictionAuditRecord, StrategyEfficacyRecord, StrategyId } from "../types";

const ids: StrategyId[] = ["freshness", "entropy", "randomness"];
function record(draw: number, a: number[], b: number[]): PredictionAuditRecord {
  return { referenceDrawNumber: draw - 1, targetDrawNumber: draw, date: null,
    numbers: [1, 2, 3, 4, 5, 6].map(number => ({ number, strategies: [
      ...(a.includes(number) ? [{ id: "freshness" as const, name: "Freshness" }] : []),
      ...(b.includes(number) ? [{ id: "entropy" as const, name: "Entropy" }] : []),
    ] })) };
}
function efficacy(audit: PredictionAuditRecord[]): StrategyEfficacyRecord[] {
  return audit.map(record => ({ referenceDrawNumber: record.referenceDrawNumber,
    targetDrawNumber: record.targetDrawNumber, actualNumbers: record.numbers.map(item => item.number), randomHits: 0,
    strategyHits: Object.fromEntries(ids.map(id => [id, record.numbers.filter(item => item.strategies.some(strategy => strategy.id === id)).length])) }));
}

describe("strategy hit statistics", () => {
  it("counts shared events separately from same-draw hits and retains zero-hit strategies", () => {
    const audit = [record(2, [1, 2], [1, 2, 3]), record(3, [1], [4]), record(4, [1], [1]), record(5, [], [])];
    const history = efficacy(audit);
    const result = strategyHitStatistics(audit, history, ids);
    const pair = result.pairs[0];
    expect(pair).toEqual({ left: "freshness", right: "entropy", sharedNumbers: 3, sharedDraws: 2, bothHitDraws: 3 });
    expect(contributingDraws(result.draws, pair, "sharedNumbers").map(draw => draw.record.targetDrawNumber)).toEqual([2, 4]);
    expect(contributingDraws(result.draws, pair, "bothHitDraws")).toHaveLength(3);
    expect(result.pairs).toHaveLength(3);
    for (const strategy of result.strategies) {
      expect(strategy.hits).toBe(history.reduce((sum, record) => sum + (record.strategyHits[strategy.id] ?? 0), 0));
      expect(strategy.buckets.reduce((sum, count) => sum + count, 0)).toBe(4);
    }
    expect(result.strategies[2]).toMatchObject({ hits: 0, hitDraws: 0, average: 0, buckets: [4, 0, 0, 0, 0, 0, 0] });
  });
  it("applies history scope to draws, totals, and relationships", () => {
    const audit = Array.from({ length: 101 }, (_, index) => record(index + 2, index === 0 ? [1] : [], index === 0 ? [1] : []));
    const result = strategyHitStatistics(audit, efficacy(audit), ids, 100);
    expect(result.draws).toHaveLength(100);
    expect(result.draws[0].record.targetDrawNumber).toBe(3);
    expect(result.strategies[0].hits).toBe(0);
    expect(result.strategies[0].buckets[0]).toBe(100);
    expect(result.pairs[0].sharedNumbers).toBe(0);
  });
  it("handles empty history without division by zero", () => {
    const result = strategyHitStatistics([], [], ids);
    expect(result.draws).toEqual([]);
    expect(result.strategies.every(strategy => strategy.average === 0 && strategy.evaluatedDraws === 0)).toBe(true);
  });
  it("does not count a missing strategy evaluation as a zero-hit draw", () => {
    const audit = [record(2, [], [])];
    const history = efficacy(audit);
    delete history[0].strategyHits.entropy;
    expect(strategyHitStatistics(audit, history, ids).strategies[1].evaluatedDraws).toBe(0);
  });
});
