import { createSSRApp } from "vue";
import { renderToString } from "@vue/server-renderer";
import { expect, it } from "vitest";
import StrategyHitStatisticsView from "./StrategyHitStatisticsView.vue";
import type { AnalysisPayload, PredictionAuditRecord } from "../types";

it("renders draw details, zero-hit strategies, relationship controls, and symmetric cells", async () => {
  const audit: PredictionAuditRecord[] = Array.from({ length: 101 }, (_, index) => ({
    referenceDrawNumber: index + 1, targetDrawNumber: index + 2, date: null,
    numbers: [1, 2, 3, 4, 5, 6].map(number => ({ number, strategies:
      number === 1 ? [{ id: "freshness", name: "Freshness" }] :
      number === 2 ? [{ id: "entropy", name: "Entropy" }] : [] })),
  }));
  const analysis = { options: { enabledStrategies: ["freshness", "entropy", "randomness"] },
    predictionAuditHistory: audit,
    strategyEfficacyHistory: audit.map(record => ({ targetDrawNumber: record.targetDrawNumber,
      strategyHits: { freshness: 1, entropy: 1, randomness: 0 } })) } as unknown as AnalysisPayload;
  const html = await renderToString(createSSRApp(StrategyHitStatisticsView, { analysis }));
  expect(html).toContain("101 evaluated draws");
  expect(html).toContain("Predicted from draw 101.");
  expect(html).toContain("Random baseline");
  expect(html).toMatch(/<option value="102"[^>]* selected>/);
  expect(html).toContain("Both hit in the same draw");
  expect(html).toContain('aria-label="Freshness and Entropy: 0"');
  expect(html).toContain('aria-label="Entropy and Freshness: 0"');
  expect(html).toContain("Shared numbers ↕");
});

it("renders an empty history", async () => {
  const analysis = { options: { enabledStrategies: [] }, predictionAuditHistory: [], strategyEfficacyHistory: [] } as unknown as AnalysisPayload;
  const html = await renderToString(createSSRApp(StrategyHitStatisticsView, { analysis }));
  expect(html).toContain("No evaluated draws available.");
  expect(html).not.toContain("NaN");
});
