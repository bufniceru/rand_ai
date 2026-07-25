<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import type { CombinedPredictionDialogData } from "./types";
import PredictionWorkspaceNavigation from "./components/PredictionWorkspaceNavigation.vue";
import CombinedPredictionGridView from "./views/CombinedPredictionGridView.vue";

const dialogData = ref<CombinedPredictionDialogData | null>(null);
const errorMessage = ref("");
let unsubscribeData: (() => void) | null = null;

onMounted(async () => {
  document.title = "Predictions — Rand AI";
  if (!window.randAiDesktop) {
    errorMessage.value = "Combined prediction is available inside the Electron application.";
    return;
  }
  unsubscribeData = window.randAiDesktop.onCombinedPredictionData((data) => {
    dialogData.value = data;
  });
  try {
    dialogData.value = await window.randAiDesktop.getCombinedPredictionData();
    if (!dialogData.value) {
      errorMessage.value = "Analyze a dataset before opening this prediction.";
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
});

onBeforeUnmount(() => {
  unsubscribeData?.();
});
</script>

<template>
  <main class="combined-prediction-dialog-shell">
    <PredictionWorkspaceNavigation v-if="!dialogData" active="predictions" />
    <CombinedPredictionGridView
      v-if="dialogData"
      :prediction-suites="dialogData.predictionSuites"
      :efficacy-history="dialogData.strategyEfficacyHistory"
    />
    <section v-else class="dialog-empty-state">
      <strong>Predictions unavailable</strong>
      <p>{{ errorMessage || "Loading precomputed predictions…" }}</p>
    </section>
  </main>
</template>
