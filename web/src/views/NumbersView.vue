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
      <div><p class="eyebrow">Number analysis</p><h2>Frequency and relationships</h2></div>
      <p>Overall appearances, sorted positions, pair co-occurrence, and binned trends.</p>
    </header>
    <article class="chart-card wide"><PlotlyChart :figure="figures.number_frequencies" /></article>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.position_frequencies" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.pair_cooccurrence" /></article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.number_trends" /></article>
    <div class="table-grid">
      <article class="table-card">
        <h3>Overall frequency</h3>
        <DataTable :table="analysis.tables.number_frequencies" />
      </article>
      <article class="table-card">
        <h3>Descriptive statistics</h3>
        <DataTable :table="analysis.tables.number_descriptive" :searchable="false" />
      </article>
    </div>
  </section>
</template>
