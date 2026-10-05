import { describe, expect, it } from "vitest";
import { drawHistoryNumberHits } from "./drawHistoryHits";
import type { DrawEditorEntry, PredictionAuditRecord, StrategyId } from "../types";

const draw: DrawEditorEntry = { index: 1, date: "2026-10-01", numbers: [1, 2, 3, 4, 5, 6] };
const ids: StrategyId[] = ["svc_recurrence_proximity_hybrid", "srph_residual_diversity_hybrid"];
const history: PredictionAuditRecord[] = [{ referenceDrawNumber: 1, targetDrawNumber: 2, date: draw.date,
  numbers: draw.numbers.map(number => ({ number, strategies: number === 1
    ? [{ id: ids[0], name: "SRPH" }, { id: ids[1], name: "SRD" }] : [] })) }];
describe("draw history number hits", () => {
  it("matches a zero-based editor index and expands strategy abbreviations", () => {
    const result = drawHistoryNumberHits(draw, 1, history, ids);
    expect(result.message).toBe("");
    expect(result.strategies.map(strategy => strategy.name)).toEqual([
      "SRPH Residual Diversity Hybrid", "SVC–Recurrence–Proximity Hybrid",
    ]);
  });
  it("filters disabled strategies and distinguishes zero hits", () => {
    expect(drawHistoryNumberHits(draw, 1, history, [ids[0]]).strategies).toHaveLength(1);
    expect(drawHistoryNumberHits(draw, 2, history, ids).message).toBe("No enabled strategy included this number in its prior Top 6.");
    expect(drawHistoryNumberHits(draw, 1, history, []).message).toBe("No strategies enabled.");
  });
  it("rejects mismatched dates, numbers, indices, and unavailable forecasts", () => {
    for (const changed of [{ ...draw, index: 2 }, { ...draw, date: "2026-10-02" }, { ...draw, numbers: [1, 2, 3, 4, 5, 7] }]) {
      expect(drawHistoryNumberHits(changed, 1, history, ids).message).toBe("No prior prediction available for this draw.");
    }
    expect(drawHistoryNumberHits(draw, 1, [], ids).message).toBe("No prior prediction available for this draw.");
  });
  it("suppresses all prior hits while analysis is stale", () => {
    expect(drawHistoryNumberHits(draw, 1, history, ids, true)).toEqual({ strategies: [], message: "Reanalyze to refresh strategy hits." });
  });
});
