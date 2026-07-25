<script setup lang="ts">
import PlotlyChart from "../components/PlotlyChart.vue";
import type { AnalysisPayload, FigureSpec } from "../types";

defineProps<{
  analysis: AnalysisPayload;
  figures: Record<string, FigureSpec>;
}>();
</script>

<template>
  <section class="workspace-view">
    <header class="view-header">
      <div><p class="eyebrow">Relationships</p><h2>Correlation matrices</h2></div>
      <p>
        {{ analysis.options.correlationMethod === "spearman"
          ? `Spearman uses a deterministic sample of ${analysis.dataset.sampleSize.toLocaleString()} draws.`
          : "Pearson uses the complete dataset." }}
      </p>
    </header>
    <article class="chart-card wide"><PlotlyChart :figure="figures.number_correlations" /></article>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.space_correlations" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.number_space_correlations" /></article>
    </div>
  </section>
</template>
