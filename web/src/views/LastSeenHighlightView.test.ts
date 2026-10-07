import { createSSRApp } from "vue";
import { renderToString } from "@vue/server-renderer";
import { beforeEach, expect, it } from "vitest";
import LastSeenHighlightView from "./LastSeenHighlightView.vue";
import { configurePossibleDrawContext, resetPossibleDrawStoreForTests, setPossibleDrawNumberState } from "../lib/possibleDrawPlans";
import type { HistoryDraw } from "../types";

const history = [1, 2].map(drawNumber => ({ drawNumber, numbers: [] })) as unknown as HistoryDraw[];
beforeEach(() => {
  resetPossibleDrawStoreForTests();
  configurePossibleDrawContext({ datasetId: "header-test", targetDrawId: "3" });
});
function render(referenceDrawOffset = 0) {
  return renderToString(createSSRApp(LastSeenHighlightView, { history, drawCount: 2, referenceDrawOffset }));
}
it("renders all 49 accessible controls and reflects shared selection changes", async () => {
  setPossibleDrawNumberState(1, "candidate");
  setPossibleDrawNumberState(2, "fixed");
  setPossibleDrawNumberState(3, "excluded");
  let html = await render();
  expect(html.match(/role="button"/g)).toHaveLength(49);
  expect(html).toContain("Number 1; Possible Draw state candidate");
  expect(html).toContain("Number 2; Possible Draw state fixed");
  expect(html).toContain("Number 3; Possible Draw state excluded");
  expect(html).toContain('aria-disabled="false"');
  setPossibleDrawNumberState(1, "neutral");
  html = await render();
  expect(html).toContain("Number 1; Possible Draw state neutral");
});
it("shows historical references as read-only while retaining shared states", async () => {
  setPossibleDrawNumberState(4, "fixed");
  const html = await render(1);
  expect(html.match(/aria-disabled="true"/g)).toHaveLength(49);
  expect(html).toContain("Number 4; Possible Draw state fixed; read-only historical reference");
  expect(html).toContain("Return to the latest draw to edit");
  expect(await render()).toContain('aria-disabled="false"');
});
it("does not offer editing for empty history", async () => {
  const html = await renderToString(createSSRApp(LastSeenHighlightView, { history: [], drawCount: 2, referenceDrawOffset: 0 }));
  expect(html.match(/aria-disabled="true"/g)).toHaveLength(49);
});
