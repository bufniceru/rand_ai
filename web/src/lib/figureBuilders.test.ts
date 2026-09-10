import { describe, expect, it } from "vitest";
import type { AnalysisPayload } from "../types";
import { buildFigures } from "./figureBuilders";

describe("gap randomness charts", () => {
  it("plots counts and rates from the same table, including a zero-hit gap", () => {
    const rows = [
      { gap: 0, hits: 12, opportunities: 55, hit_rate: 1200 / 55,
        hit_percentage: 100, expected_hits: 330 / 49, hit_difference: 12 - 330 / 49,
        expected_hit_rate: 600 / 49, hit_rate_difference_pp: 1200 / 55 - 600 / 49 },
      { gap: 1, hits: 0, opportunities: 43, hit_rate: 0,
        hit_percentage: 0, expected_hits: 258 / 49, hit_difference: -258 / 49,
        expected_hit_rate: 600 / 49, hit_rate_difference_pp: -600 / 49 },
    ];
    const analysis = {
      options: { enabledReports: ["gaps"] },
      tables: { freshness_gap_distribution: { columns: Object.keys(rows[0]), rows } },
    } as unknown as AnalysisPayload;
    const figures = buildFigures(analysis);
    const counts = figures.freshness_gap_distribution.data;
    const rates = figures.freshness_gap_hit_rate.data;
    expect(counts[0].y).toEqual([12, 0]);
    expect(counts[1].y).toEqual([330 / 49, 258 / 49]);
    expect(rates[0].y).toEqual([1200 / 55, 0]);
    expect(rates[1].y).toEqual([600 / 49, 600 / 49]);
    expect(rates[0].customdata).toEqual([
      [55, 12, 1200 / 55 - 600 / 49], [43, 0, -600 / 49],
    ]);
    for (const trace of [...counts, ...rates]) {
      expect(trace.x).toEqual([0, 1]);
      expect(trace.hovertemplate).toContain("Opportunities");
    }
  });
});
