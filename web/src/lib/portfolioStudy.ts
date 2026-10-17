/** Research-only comparisons; random streams never depend on target outcomes. */
import { runPortfolioBacktest } from "./drawPortfolioBacktest";
import { verifyTripleCoverage, type PortfolioObjective } from "./drawPortfolio";
import type { PortfolioBacktestData } from "../types";

export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomTickets(pool: readonly number[], count: number, random: () => number, fixed: readonly number[] = []): number[][] {
  const values = [...new Set(pool)];
  if (!Number.isInteger(count) || values.length < 6 || fixed.length > 6 || new Set(fixed).size !== fixed.length || fixed.some((n) => !values.includes(n)))
    throw new Error("Invalid random portfolio universe");
  const free = values.filter((n) => !fixed.includes(n));
  let capacity = 1;
  for (let i = 1; i <= 6 - fixed.length; i++) capacity = capacity * (free.length - i + 1) / i;
  if (count < 0 || count > Math.round(capacity)) throw new Error("Insufficient unique random tickets");
  const tickets = new Map<string, number[]>();
  while (tickets.size < count) {
    const shuffled = [...free];
    for (let i = 0; i < 6 - fixed.length; i++) {
      const j = i + Math.floor(random() * (shuffled.length - i));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const ticket = [...fixed, ...shuffled.slice(0, 6 - fixed.length)].sort((a, b) => a - b);
    tickets.set(ticket.join(","), ticket);
  }
  return [...tickets.values()];
}

export function ticketMetrics(tickets: readonly number[][], actual: readonly number[], pool: readonly number[]) {
  const winning = new Set(actual);
  const bestHits = tickets.reduce((best, ticket) => Math.max(best, ticket.filter((n) => winning.has(n)).length), 0);
  const universe = new Set(pool);
  const triples = new Set<string>();
  for (const ticket of tickets) {
    const values = ticket.filter((n) => universe.has(n)).slice().sort((a, b) => a - b);
    for (let i = 0; i < values.length - 2; i++)
      for (let j = i + 1; j < values.length - 1; j++)
        for (let k = j + 1; k < values.length; k++) triples.add(`${values[i]},${values[j]},${values[k]}`);
  }
  const totalTriples = pool.length * (pool.length - 1) * (pool.length - 2) / 6;
  return { bestHits, threePlus: Number(bestHits >= 3), generatedTickets: tickets.length,
    coveredTriples: triples.size, totalTriples, tripleCoverage: totalTriples > 0 ? triples.size / totalTriples : 0 };
}

export interface StudyOptions {
  budget: number;
  seed: number;
  initialHistory: number;
  maxTargets?: number;
  randomReplicates: number;
}

export function runPortfolioStudy(data: PortfolioBacktestData, options: StudyOptions) {
  if (!Number.isInteger(options.randomReplicates) || options.randomReplicates < 1) throw new Error("Positive random replicate count required");
  for (let i = 0; i < data.records.length; i++) {
    const record = data.records[i];
    if (record.targetDrawNumber !== record.referenceDrawNumber + 1 ||
        (i > 0 && record.referenceDrawNumber <= data.records[i - 1].referenceDrawNumber))
      throw new Error("Portfolio study records must be chronological next-draw forecasts");
  }
  const allNumbers = Array.from({ length: 49 }, (_, i) => i + 1);
  const rows: Record<string, unknown>[] = [];
  const uncovered: Record<string, unknown>[] = [];
  for (const objective of ["blended", "coverage_first"] as PortfolioObjective[]) {
    const started = performance.now();
    runPortfolioBacktest(data, options.budget, undefined, {
      objective, initialHistory: options.initialHistory, maxTargets: options.maxTargets,
      onPortfolio(portfolio, record, generationMs) {
        const pool = portfolio.pool.map((entry) => entry.number);
        const tickets = portfolio.draws.map((entry) => entry.numbers);
        const exact = verifyTripleCoverage(pool, tickets);
        const common = { objective, reference: record.referenceDrawNumber, target: record.targetDrawNumber, budget: options.budget };
        rows.push({ ...common, method: objective, date: record.date, actualNumbers: record.actualNumbers, generationMs, ...ticketMetrics(tickets, record.actualNumbers, pool),
          weightedTripleCoverage: portfolio.metrics.tripleCoverage, pool, tickets,
          uncoveredTriples: exact.uncoveredCount, guaranteed: exact.guaranteed, guaranteeCondition: exact.condition });
        uncovered.push({ ...common, pool: [...pool].sort((a, b) => a - b), total: exact.total, covered: exact.covered, uncoveredCount: exact.uncoveredCount, uncovered: exact.uncovered });
        // Reinitialize by target: both objectives receive paired baseline portfolios.
        const random = seededRandom((options.seed ^ Math.imul(record.targetDrawNumber, 2654435761)) >>> 0);
        for (let replicate = 0; replicate < options.randomReplicates; replicate++) {
          for (const method of ["random_global", "random_pool"] as const) {
            const universe = method === "random_global" ? allNumbers : pool;
            const randomPortfolio = randomTickets(universe, tickets.length, random,
              method === "random_pool" && portfolio.metadata.mode === "guided" ? portfolio.metadata.fixedNumbers : []);
            rows.push({ ...common, method, replicate,
              ...ticketMetrics(randomPortfolio, record.actualNumbers, universe),
              selectedPoolTripleCoverage: ticketMetrics(randomPortfolio, record.actualNumbers, pool).tripleCoverage });
          }
        }
      },
    });
    rows.push({ objective, method: "runtime", budget: options.budget, durationMs: performance.now() - started });
  }
  return { rows, uncovered };
}
