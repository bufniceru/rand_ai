import { createSSRApp } from "vue";
import { renderToString } from "@vue/server-renderer";
import { expect, it } from "vitest";
import type { AnalysisPayload, FirstNumberDirectionAnalysis } from "../types";
import FirstNumberDirectionView from "./FirstNumberDirectionView.vue";

const render = (data: FirstNumberDirectionAnalysis | null) => renderToString(createSSRApp(FirstNumberDirectionView, {
  analysis: { firstNumberDirection: data } as AnalysisPayload,
}));

it("renders an empty state", async () => {
  expect(await render(null)).toContain("Load at least one draw");
});

it("shows exact probabilities, reference draw and pending evidence", async () => {
  const data: FirstNumberDirectionAnalysis = {
    protocol: {}, datasetFingerprint: "abc", blocks: [], history: [], evaluation: {},
    forecast: { referenceDrawNumber: 1, referenceDate: null, referenceMinimum: 2,
      targetDrawNumber: 2, probabilities: [.12, .11, .77], baselineProbabilities: [.12, .11, .77],
      predictedDirection: "higher", activeModel: "baseline", models: {},
      activationEvidence: { scoredCount: 0, eligible: false, candidates: {} } },
  };
  const html = await render(data);
  expect(html).toContain("Following draw 1");
  expect(html).toContain("Next minimum: higher");
  expect(html).toContain("77.00%");
  expect(html).toContain("activation pending");
  expect(html.match(/<progress /g)).toHaveLength(3);
  expect(html).not.toContain("NaN");
});

it("renders gated evidence, equality history and evaluation", async () => {
  const forecast = { referenceDrawNumber: 120, referenceDate: "2026-01-01", referenceMinimum: 2,
    targetDrawNumber: 121, probabilities: [.2, .6, .2], baselineProbabilities: [.12, .11, .77],
    predictedDirection: "equal" as const, activeModel: "historical", models: {},
    activationEvidence: { scoredCount: 200, eligible: true, candidates: {
      historical: { improvement: .1, lower95: .02, pValue: .001, holmPValue: .003, supported: true },
    } } };
  const metric = { count: 201, logLoss: .5, brier: .3, accuracy: .7, confusionMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], recall: [.7, null, .5] };
  const data: FirstNumberDirectionAnalysis = { protocol: {}, datasetFingerprint: "abc",
    forecast, evaluation: { gated: metric }, blocks: [{ gated: metric }],
    history: [{ ...forecast, actualDate: "2026-01-02", actualMinimum: 2, actualDirection: "equal" }] };
  const original = JSON.stringify(data);
  const html = await render(data);
  expect(html).toContain("historical");
  expect(html).toContain("0.0030");
  expect(html).toContain("2026-01-02");
  expect(html).toContain("0.5000");
  expect(html).toContain("Block 1");
  expect(html).not.toContain("activation pending");
  expect(JSON.stringify(data)).toBe(original);
});
