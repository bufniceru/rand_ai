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
      <div><p class="eyebrow">Circular spaces</p><h2>Distance structure</h2></div>
      <p>Every draw has six circular spaces whose values sum to 43.</p>
    </header>
    <article class="chart-card wide"><PlotlyChart :figure="figures.distance_frequencies" /></article>
    <div class="chart-grid">
      <article v-for="position in 6" :key="position" class="chart-card">
        <PlotlyChart :figure="figures[`dist${position}_frequencies`]" />
      </article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.space_frequencies" /></article>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.space_box_plots" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.space_extremes" /></article>
    </div>
    <article class="table-card">
      <h3>Space descriptive statistics</h3>
      <DataTable :table="analysis.tables.space_descriptive" :searchable="false" />
    </article>
  </section>
</template>
