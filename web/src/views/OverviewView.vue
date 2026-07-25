<script setup lang="ts">
import { computed } from "vue";
import DataTable from "../components/DataTable.vue";
import PlotlyChart from "../components/PlotlyChart.vue";
import { summaryLookup } from "../lib/figureBuilders";
import type { AnalysisPayload, FigureSpec } from "../types";

const props = defineProps<{
  analysis: AnalysisPayload;
  figures: Record<string, FigureSpec>;
}>();

const summary = computed(() => summaryLookup(props.analysis));
</script>

<template>
  <section class="workspace-view">
    <header class="view-header">
      <div>
        <p class="eyebrow">Dataset overview</p>
        <h2>Draw history at a glance</h2>
      </div>
      <p>Frequency, composition, and combination coverage for the active dataset.</p>
    </header>
    <div class="metric-grid">
      <article class="metric-card">
        <span>Draws</span>
        <strong>{{ analysis.dataset.drawCount.toLocaleString() }}</strong>
      </article>
      <article class="metric-card">
        <span>Unique combinations</span>
        <strong>{{ (summary.get("unique_combinations") ?? 0).toLocaleString() }}</strong>
      </article>
      <article class="metric-card">
        <span>Repeated draws</span>
        <strong>{{ (summary.get("repeated_draws") ?? 0).toLocaleString() }}</strong>
      </article>
      <article class="metric-card">
        <span>Heavy-analysis sample</span>
        <strong>{{ analysis.dataset.sampleSize.toLocaleString() }}</strong>
      </article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.number_frequencies" /></article>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.draw_sum_distribution" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.draw_composition" /></article>
    </div>
    <article class="table-card">
      <h3>Overview statistics</h3>
      <DataTable :table="analysis.tables.summary" :searchable="false" />
    </article>
  </section>
</template>
