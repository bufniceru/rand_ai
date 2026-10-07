import { createSSRApp } from "vue";
import { renderToString } from "@vue/server-renderer";
import { expect, it } from "vitest";
import type { AnalysisPayload } from "../types";
import StrategyEffectivenessView from "./StrategyEffectivenessView.vue";

it("renders zero hits and separate evaluation counts without double-counted aggregate totals", async () => {
  const analysis = { strategyEfficacyHistory: [
    { targetDrawNumber: 121, strategyHits: { svc: 0 } },
    { targetDrawNumber: 122, strategyHits: { svc: 2, tbl: 3 } },
  ] } as unknown as AnalysisPayload;
  const html = await renderToString(createSSRApp(StrategyEffectivenessView, { analysis }));
  expect(html).toContain("Exactly 0");
  expect(html).toContain("Evaluated draws");
  expect(html).toContain("Random expectation");
  expect(html).not.toContain("All correct implications");
  expect(html).not.toContain('class="all-strategies-summary"');
  expect(html).toMatch(/<td>1<\/td>\s*<td>3<\/td>\s*<td>3\.000<\/td>/);
  expect(html).not.toContain("NaN");
});

it("keeps grouped original methods accessible and preserves their source records", async () => {
  const analysis = { strategyEfficacyHistory: [
    { targetDrawNumber: 121, strategyHits: { markov100: 1, mkgsv: 1 } },
    { targetDrawNumber: 122, strategyHits: { markov100: 2, mkgsv: 2 } },
  ] } as unknown as AnalysisPayload;
  const original = JSON.stringify(analysis);
  const html = await renderToString(createSSRApp(StrategyEffectivenessView, { analysis }));
  expect(html).toContain("Similar methods from the study");
  expect(html).toContain("1 individual controls");
  expect(html).toContain("Markov Gap-Space Vector");
  expect(html.match(/type="checkbox" checked/g)).toHaveLength(1);
  expect(JSON.stringify(analysis)).toBe(original);
});
