import { createSSRApp } from "vue";
import { renderToString } from "@vue/server-renderer";
import { beforeEach, expect, it } from "vitest";
import CircleNumberSelectorView from "./CircleNumberSelectorView.vue";
import LastSeenHighlightView from "./LastSeenHighlightView.vue";
import { circleNumbers, NUMBER_BALL_RADIUS, CIRCLE_RADIUS } from "../lib/circleNumbers";
import { configurePossibleDrawContext, resetPossibleDrawStoreForTests, setPossibleDrawNumberState } from "../lib/possibleDrawPlans";
import type { HistoryDraw } from "../types";

const history = [{ drawNumber: 2, date: null, numbers: [] }] as HistoryDraw[];
beforeEach(() => {
  resetPossibleDrawStoreForTests();
  configurePossibleDrawContext({ datasetId: "circle-test", targetDrawId: "3" });
});
function render(draws = history, targetDrawNumber: number | null = 3) {
  return renderToString(createSSRApp(CircleNumberSelectorView, { history: draws, targetDrawNumber }));
}
it("renders 49 clockwise controls at the header ball size with no overlapping balls", async () => {
  const html = await render();
  expect(html.match(/role="button"/g)).toHaveLength(49);
  expect(html.match(/tabindex="0"/g)).toHaveLength(49);
  expect(html.match(/r="13.5"/g)).toHaveLength(49);
  expect(html).toContain('width="580" height="580"');
  expect(html).toContain("Target draw 3");
  expect(circleNumbers[0].x).toBeCloseTo(290);
  expect(circleNumbers[0].y).toBeCloseTo(30);
  expect(circleNumbers[1].x).toBeGreaterThan(290);
  expect(circleNumbers[1].y).toBeGreaterThan(30);
  for (const point of circleNumbers) {
    expect(Math.hypot(point.x - 290, point.y - 290)).toBeCloseTo(CIRCLE_RADIUS);
    for (const other of circleNumbers) {
      if (point.number === other.number) continue;
      expect(Math.hypot(point.x - other.x, point.y - other.y)).toBeGreaterThan(2 * NUMBER_BALL_RADIUS);
    }
  }
});
it("reflects the same plan as Last Seen and isolates dataset contexts", async () => {
  setPossibleDrawNumberState(1, "candidate");
  setPossibleDrawNumberState(2, "fixed");
  setPossibleDrawNumberState(3, "excluded");
  const circle = await render();
  const lastSeen = await renderToString(createSSRApp(LastSeenHighlightView, { history, drawCount: 1, referenceDrawOffset: 0 }));
  for (const [number, state] of [[1, "candidate"], [2, "fixed"], [3, "excluded"]]) {
    const label = `Number ${number}; Possible Draw state ${state}`;
    expect(circle).toContain(label);
    expect(lastSeen).toContain(label);
  }
  configurePossibleDrawContext({ datasetId: "other", targetDrawId: "3" });
  expect(await render()).toContain("Number 2; Possible Draw state neutral");
  configurePossibleDrawContext({ datasetId: "circle-test", targetDrawId: "3" });
  expect(await render()).toContain("Number 2; Possible Draw state fixed");
});
it("disables editing without history or a target", async () => {
  expect((await render([])).match(/aria-disabled="true"/g)).toHaveLength(49);
  expect((await render(history, null)).match(/aria-disabled="true"/g)).toHaveLength(49);
  expect(await render([])).toContain("Load a dataset with history");
});
