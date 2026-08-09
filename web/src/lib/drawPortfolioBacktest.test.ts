import { describe, expect, it } from "vitest";

import type {
  PortfolioBacktestData,
  PortfolioBacktestRecord,
  StrategyId,
} from "../types";
import {
  portfolioBacktestCacheKey,
  portfolioHistoryStatistics,
  runPortfolioBacktest,
} from "./drawPortfolioBacktest";

function ranking(offset: number): number[] {
  return Array.from({ length: 49 }, (_value, index) => ((index + offset) % 49) + 1);
}

function record(
  reference: number,
  actualNumbers: number[],
): PortfolioBacktestRecord {
  return {
    referenceDrawNumber: reference,
    targetDrawNumber: reference + 1,
    date: `2026-01-${String(reference + 1).padStart(2, "0")}`,
    actualNumbers,
    strategies: [
      { id: "freshness" as StrategyId, ranking: ranking(0) },
      { id: "entropy" as StrategyId, ranking: ranking(12) },
    ],
  };
}

function data(): PortfolioBacktestData {
  return {
    cacheKey: "a".repeat(64),
    strategyIds: ["freshness", "entropy"],
    draws: [
      { date: "2026-01-01", numbers: [1, 2, 3, 4, 5, 6] },
      { date: "2026-01-02", numbers: [7, 8, 9, 10, 11, 12] },
      { date: "2026-01-03", numbers: [13, 14, 15, 16, 17, 18] },
      { date: "2026-01-04", numbers: [19, 20, 21, 22, 23, 24] },
    ],
    records: [
      record(1, [7, 8, 9, 10, 11, 12]),
      record(2, [13, 14, 15, 16, 17, 18]),
      record(3, [19, 20, 21, 22, 23, 24]),
    ],
  };
}

describe("portfolio walk-forward backtest", () => {
  it("builds exact and cumulative buckets from best-ticket results", () => {
    const progress: number[] = [];
    const result = runPortfolioBacktest(data(), 2, (update) => {
      progress.push(update.processed ?? 0);
    });

    expect(result.portfolioSize).toBe(2);
    expect(result.evaluatedTargets).toBe(3);
    expect(result.audit).toHaveLength(3);
    expect(result.buckets.reduce((total, bucket) => total + bucket.exactCount, 0)).toBe(3);
    expect(result.buckets[0].atLeastCount).toBe(3);
    expect(result.buckets[6].atLeastCount).toBe(result.buckets[6].exactCount);
    expect(progress).toEqual([1, 2, 3]);
    for (let hits = 0; hits < 6; hits += 1) {
      expect(result.buckets[hits].atLeastCount).toBeGreaterThanOrEqual(
        result.buckets[hits + 1].atLeastCount,
      );
    }
  });

  it("does not allow future draws or the current target outcome into its portfolio", () => {
    const original = data();
    const changed = structuredClone(original);
    changed.draws[2].numbers = [30, 31, 32, 33, 34, 35];
    changed.records[0].actualNumbers = [40, 41, 42, 43, 44, 45];

    const first = runPortfolioBacktest(original, 3).audit[0];
    const changedFirst = runPortfolioBacktest(changed, 3).audit[0];

    expect(changedFirst.bestTicket).toEqual(first.bestTicket);
  });

  it("chooses the lexicographically first ticket when best hits tie", () => {
    const result = runPortfolioBacktest(data(), 5);
    for (const row of result.audit) {
      expect(row.bestTicket).toEqual([...row.bestTicket].sort((left, right) => left - right));
      expect(row.tiedBestCount).toBeGreaterThanOrEqual(1);
    }
  });

  it("creates stable versioned persistent-cache keys", () => {
    expect(portfolioBacktestCacheKey("a".repeat(64), 10)).toBe(
      `${"a".repeat(64)}-v3-p10`,
    );
  });

  it("summarizes the maximum and average best result across history", () => {
    const statistics = portfolioHistoryStatistics({
      portfolioSize: 12,
      audit: [
        { ...record(1, [1, 2, 3, 4, 5, 6]), bestTicket: [], bestHits: 2, tiedBestCount: 1 },
        { ...record(2, [1, 2, 3, 4, 5, 6]), bestTicket: [], bestHits: 4, tiedBestCount: 1 },
        { ...record(3, [1, 2, 3, 4, 5, 6]), bestTicket: [], bestHits: 4, tiedBestCount: 2 },
      ],
    });

    expect(statistics).toEqual({
      evaluatedTargets: 3,
      portfolioSize: 12,
      maximumHits: 4,
      maximumHitTargets: 2,
      maximumHitRate: 2 / 3,
      averageBestHits: 10 / 3,
    });
  });

  it("returns zero statistics before a historical simulation exists", () => {
    expect(portfolioHistoryStatistics(null)).toEqual({
      evaluatedTargets: 0,
      portfolioSize: 0,
      maximumHits: 0,
      maximumHitTargets: 0,
      maximumHitRate: 0,
      averageBestHits: 0,
    });
  });

  it("supports and clamps historical portfolios at 100 tickets", () => {
    const emptyHistory = { ...data(), records: [] };
    expect(runPortfolioBacktest(emptyHistory, 100).portfolioSize).toBe(100);
    expect(runPortfolioBacktest(emptyHistory, 101).portfolioSize).toBe(100);
  });
});
