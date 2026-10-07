<script setup lang="ts">
import DataTable from "../components/DataTable.vue";
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
      <div><p class="eyebrow">Number analysis</p><h2>Positions and relationships</h2></div>
      <p>Historical positions, pair co-occurrence, and trends. Predictive usefulness requires a separate comparison of future number hits.</p>
    </header>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.position_frequencies" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.pair_cooccurrence" /></article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.number_trends" /></article>
    <details class="table-card">
      <summary>Descriptive statistics</summary>
      <DataTable :table="analysis.tables.number_descriptive" :searchable="false" />
    </details>
  </section>
</template>
