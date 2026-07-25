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
      <div><p class="eyebrow">Diagnostics</p><h2>Randomness checks</h2></div>
      <p>These diagnostics describe this dataset; they do not predict future draws.</p>
    </header>
    <aside class="warning-banner">
      A fair lottery does not make any number more likely because of these historical results.
    </aside>
    <article class="chart-card wide"><PlotlyChart :figure="figures.matching_pairs" /></article>
    <article class="table-card">
      <h3>Randomness diagnostics</h3>
      <DataTable :table="analysis.tables.randomness_diagnostics" :searchable="false" />
    </article>
  </section>
</template>
