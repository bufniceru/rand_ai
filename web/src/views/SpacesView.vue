<script setup lang="ts">
import { ref } from "vue";
import DataTable from "../components/DataTable.vue";
import PlotlyChart from "../components/PlotlyChart.vue";
import type { AnalysisPayload, FigureSpec } from "../types";

defineProps<{
  analysis: AnalysisPayload;
  figures: Record<string, FigureSpec>;
}>();
const spaceDetailsOpen = ref(false);
</script>

<template>
  <section class="workspace-view">
    <header class="view-header">
      <div><p class="eyebrow">Circular spaces</p><h2>Distance structure</h2></div>
      <p>Historical structure: every draw has six circular spaces whose values sum to 43. These distributions do not establish future number hits.</p>
    </header>
    <article class="chart-card wide"><PlotlyChart :figure="figures.distance_frequencies" /></article>
    <details class="table-card" @toggle="spaceDetailsOpen = ($event.target as HTMLDetailsElement).open">
      <summary>Space counts by position · charts and combined heatmap</summary>
      <p>These charts show the same position counts in separate and combined views.</p>
      <div v-if="spaceDetailsOpen" class="chart-grid">
        <article v-for="position in 6" :key="position" class="chart-card">
          <PlotlyChart :figure="figures[`dist${position}_frequencies`]" />
        </article>
      </div>
      <article v-if="spaceDetailsOpen" class="chart-card wide"><PlotlyChart :figure="figures.space_frequencies" /></article>
    </details>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.space_box_plots" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.space_extremes" /></article>
    </div>
    <details class="table-card">
      <summary>Space descriptive statistics</summary>
      <DataTable :table="analysis.tables.space_descriptive" :searchable="false" />
    </details>
  </section>
</template>
