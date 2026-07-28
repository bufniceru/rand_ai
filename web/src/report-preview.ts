import { createApp, h, nextTick, ref } from "vue";
import LatestDrawComparisonView from "./views/LatestDrawComparisonView.vue";
import type { AnalysisPayload } from "./types";
import "./styles.css";

declare global {
  interface Window {
    renderDrawComparisonPreview?: (
      payload: AnalysisPayload,
    ) => Promise<void>;
  }
}

const analysis = ref<AnalysisPayload | null>(null);

window.renderDrawComparisonPreview = async (payload: AnalysisPayload) => {
  analysis.value = payload;
  await nextTick();
  document.documentElement.dataset.reportReady = "true";
};

createApp({
  setup() {
    return () =>
      analysis.value
        ? h(LatestDrawComparisonView, { analysis: analysis.value })
        : h("p", "Preparing report preview...");
  },
}).mount("#app");
