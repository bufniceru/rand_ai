import { createSSRApp, ref } from "vue";
import { renderToString } from "@vue/server-renderer";
import { expect, it } from "vitest";
import type { AnalysisPayload, DirectionForecast, FirstNumberDirectionAnalysis, PositionDirectionAnalysis } from "../types";
import { useDirectionSelection } from "../lib/numberDirections";
import FirstNumberDirectionView from "./FirstNumberDirectionView.vue";

const render = (data: FirstNumberDirectionAnalysis | null) => renderToString(createSSRApp(FirstNumberDirectionView, {
  analysis: { firstNumberDirection: data } as AnalysisPayload,
}));
const metrics = { count: 201, logLoss: .5, brier: .3, accuracy: .7,
  confusionMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], recall: [.7, null, .5] };
function fixture(eligible = false): FirstNumberDirectionAnalysis {
  return { schemaVersion: 2, protocol: {}, datasetFingerprint: "abc", positions: Array.from({length: 6}, (_, i): PositionDirectionAnalysis => {
    const forecast: DirectionForecast = { referenceDrawNumber: 120, referenceDate: "2026-01-01", referenceValue: 2 + i * 8,
      targetDrawNumber: 121, probabilities: [.2, .6, .2], baselineProbabilities: [.12, .11, .77],
      predictedDirection: "equal" as const, activeModel: eligible ? "historical" : "baseline", models: {},
      activationEvidence: { scoredCount: eligible ? 200 : 0, eligible, candidates: eligible ? {
        historical: { improvement: .1, lower95: .02, pValue: .001, holmPValue: .003, supported: true },
      } : {} } };
    return { position: i + 1, protocol: {}, datasetFingerprint: "abc", forecast,
      evaluation: { gated: { ...metrics, logLoss: .5 + i } }, blocks: [{ gated: metrics }],
      history: [{ ...forecast, actualDate: `2026-01-${String(i + 2).padStart(2, "0")}`, actualValue: 2 + i * 8, actualDirection: "equal" }] };
  }) };
}

it("renders empty and zero-draw states", async () => {
  expect(await render(null)).toContain("Load at least one draw");
  const data = fixture();
  data.positions.forEach(item => { item.forecast = null; });
  expect(await render(data)).toContain("Load at least one draw");
});

it("shows six cards with unique probability bars, baseline and pending evidence", async () => {
  const html = await render(fixture());
  for (const label of ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]) {
    expect(html).toContain(`aria-label="${label} number forecast"`);
  }
  expect(html).toContain("Following draw 120");
  expect(html).toContain("77.00%");
  expect(html).toContain("activation pending");
  expect(html.match(/<progress /g)).toHaveLength(18);
  expect(new Set([...html.matchAll(/id="(direction-[^"]+)"/g)].map(match => match[1])).size).toBe(18);
  expect(html).toContain("First number details");
  expect(html).not.toContain("NaN");
});

it("renders gated evidence and only the selected position history", async () => {
  const data = fixture(true);
  const original = JSON.stringify(data);
  const html = await render(data);
  expect(html).toContain("historical");
  expect(html).toContain("0.0030");
  expect(html).toContain("2026-01-02");
  expect(html).not.toContain("2026-01-07");
  expect(html).toContain("0.5000");
  expect(html).toContain("Block 1");
  expect(html).not.toContain("activation pending");
  expect(JSON.stringify(data)).toBe(original);
});

it("switches all detail data to the selected position and updates on dataset replacement", () => {
  const report = ref<FirstNumberDirectionAnalysis | null>(fixture(true));
  const { selectedPosition, data } = useDirectionSelection(report);
  expect(data.value?.position).toBe(1);
  selectedPosition.value = 6;
  expect(data.value?.forecast?.referenceValue).toBe(42);
  expect(data.value?.evaluation.gated.logLoss).toBe(5.5);
  expect(data.value?.history[0].actualDate).toBe("2026-01-07");
  expect(data.value?.forecast?.activationEvidence.eligible).toBe(true);
  report.value = fixture(false);
  expect(data.value?.position).toBe(6);
  expect(data.value?.forecast?.activationEvidence.eligible).toBe(false);
  report.value = null;
  expect(data.value).toBeUndefined();
});
