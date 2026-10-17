import { describe, expect, it } from "vitest";
import { randomTickets, runPortfolioStudy, seededRandom, ticketMetrics } from "./portfolioStudy";
import type { PortfolioBacktestData } from "../types";

function history(): PortfolioBacktestData {
  const ranking = Array.from({ length: 49 }, (_, i) => i + 1);
  return { cacheKey: "fixture", strategyIds: ["freshness"],
    draws: [{ date: null, numbers: [1, 2, 3, 4, 5, 6] }, { date: null, numbers: [7, 8, 9, 10, 11, 12] }, { date: null, numbers: [13, 14, 15, 16, 17, 18] }],
    records: [1, 2].map((reference) => ({ referenceDrawNumber: reference, targetDrawNumber: reference + 1,
      date: null, actualNumbers: [7, 8, 9, 10, 11, 12], strategies: [{ id: "freshness", ranking }] })) };
}

describe("portfolio research", () => {
  it("samples distinct eligible tickets reproducibly and respects fixed numbers", () => {
    const pool = [1, 2, 3, 4, 5, 6, 7];
    const tickets = randomTickets(pool, 7, seededRandom(1));
    expect(new Set(tickets.map((ticket) => ticket.join())).size).toBe(7);
    expect(randomTickets(pool, 7, seededRandom(1))).toEqual(tickets);
    expect(randomTickets(pool, 3, seededRandom(2), [1, 2]).every((ticket) => ticket.includes(1) && ticket.includes(2))).toBe(true);
    expect(() => randomTickets(pool, 8, seededRandom(1))).toThrow("Insufficient");
    expect(() => randomTickets(pool, 1, seededRandom(1), [49])).toThrow("Invalid");
    expect(ticketMetrics([[1, 2, 3, 4, 5, 6]], [1, 2, 3, 7, 8, 9], pool)).toMatchObject({ bestHits: 3, threePlus: 1, coveredTriples: 20, totalTriples: 35 });
  });

  it("pairs random baselines at equal budgets and isolates future outcomes", () => {
    const options = { budget: 2, seed: 5, initialHistory: 1, maxTargets: 2, randomReplicates: 2 };
    const data = history();
    const first = runPortfolioStudy(data, options);
    const changed = history();
    changed.records[1].actualNumbers = [20, 21, 22, 23, 24, 25];
    changed.draws[2].numbers = [20, 21, 22, 23, 24, 25];
    const second = runPortfolioStudy(changed, options);
    const withoutTiming = (rows: Record<string, unknown>[]) => rows.filter((r) => r.target === 2).map(({ generationMs: _time, ...row }) => row);
    expect(withoutTiming(first.rows)).toEqual(withoutTiming(second.rows));
    expect(first.rows.filter((r) => r.method !== "runtime").every((r) => r.generatedTickets === 2)).toBe(true);
    const randomRows = first.rows.filter((r) => r.objective === "blended" && r.method === "random_pool");
    expect(randomRows).toHaveLength(4);
    expect(first.rows.filter((r) => r.method === "coverage_first").map((r) => r.pool)).toEqual(first.rows.filter((r) => r.method === "blended").map((r) => r.pool));
    expect(() => runPortfolioStudy(data, { ...options, randomReplicates: 0 })).toThrow();
    data.records.reverse();
    expect(() => runPortfolioStudy(data, options)).toThrow("chronological");
  });
});
