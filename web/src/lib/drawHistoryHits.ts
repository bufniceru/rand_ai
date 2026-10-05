import type { DrawEditorEntry, PredictionAuditRecord, StrategyEfficacyRecord, StrategyId } from "../types";
import { strategyNames } from "./strategyNames";

export function drawHistoryNumberHits(
  draw: DrawEditorEntry, number: number, history: readonly PredictionAuditRecord[],
  enabled: readonly StrategyId[], stale = false,
  efficacy: readonly StrategyEfficacyRecord[] = [],
) {
  const empty = (message: string) => ({ message, strategies: [] as { id: StrategyId; name: string; averageHits: number | null }[] });
  if (stale) return empty("Reanalyze to refresh strategy hits.");
  if (!enabled.length) return empty("No strategies enabled.");
  const numbers = [...draw.numbers].sort((a, b) => a - b);
  const record = history.find(record => record.targetDrawNumber === draw.index + 1
    && record.date === draw.date
    && record.numbers.length === numbers.length
    && record.numbers.map(item => item.number).sort((a, b) => a - b).every((value, index) => value === numbers[index]));
  const item = record?.numbers.find(item => item.number === number);
  if (!item) return empty("No prior prediction available for this draw.");
  const ids = [...new Set(item.strategies.map(strategy => strategy.id))].filter(id => enabled.includes(id));
  const prior = efficacy.filter(entry => entry.targetDrawNumber < record!.targetDrawNumber);
  const strategies = ids.map(id => {
    const counts = prior.flatMap(entry => entry.strategyHits[id] === undefined ? [] : [entry.strategyHits[id]!]);
    return { id, name: strategyNames[id] ?? item.strategies.find(strategy => strategy.id === id)!.name,
      averageHits: counts.length ? counts.reduce((sum, hits) => sum + hits, 0) / counts.length : null };
  }).sort((a, b) => (b.averageHits ?? -1) - (a.averageHits ?? -1) || a.name.localeCompare(b.name));
  return { strategies, message: strategies.length ? "" : "No enabled strategy included this number in its prior Top 6." };
}
