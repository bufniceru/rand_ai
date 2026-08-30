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
      <p>Sorted positions, pair co-occurrence, binned trends, and descriptive statistics.</p>
    </header>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.position_frequencies" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.pair_cooccurrence" /></article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.number_trends" /></article>
    <article class="table-card">
      <h3>Descriptive statistics</h3>
      <DataTable :table="analysis.tables.number_descriptive" :searchable="false" />
    </article>
  </section>
</template>
