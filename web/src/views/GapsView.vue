<script setup lang="ts">
import { computed } from "vue";
import DataTable from "../components/DataTable.vue";
import PlotlyChart from "../components/PlotlyChart.vue";
import type { AnalysisPayload, FigureSpec, TableRow } from "../types";

const props = defineProps<{
  analysis: AnalysisPayload;
  figures: Record<string, FigureSpec>;
}>();

const rows = computed(
  () => props.analysis.tables.freshness_gap_distribution?.rows ?? [],
);
const maximumGap = computed(() =>
  rows.value.reduce(
    (maximum: number, row: TableRow) => Math.max(maximum, Number(row.gap ?? 0)),
    0,
  ),
);
const totalHits = computed(() =>
  rows.value.reduce(
    (total: number, row: TableRow) => total + Number(row.hits ?? 0),
    0,
  ),
);
const mostFrequentGap = computed(() =>
  rows.value.reduce<TableRow | null>(
    (best, row) =>
      best === null || Number(row.hits ?? 0) > Number(best.hits ?? 0)
        ? row
        : best,
    null,
  ),
);
</script>

<template>
  <section class="workspace-view">
    <header class="view-header">
      <div>
        <p class="eyebrow">Freshness statistics</p>
        <h2>Gap hit distribution</h2>
      </div>
      <p>
        How many historical number hits occurred at each exact pre-draw gap.
        Gap 0 means the number also appeared in the immediately preceding draw.
        Gap 5 means five intervening draws. Before a number's first appearance,
        its gap counts draws since the dataset started.
      </p>
    </header>

    <div class="metric-grid">
      <article class="metric-card">
        <span>Gap range</span>
        <strong>0–{{ maximumGap }}</strong>
      </article>
      <article class="metric-card">
        <span>Total number hits</span>
        <strong>{{ totalHits.toLocaleString() }}</strong>
      </article>
      <article class="metric-card">
        <span>Most frequent gap</span>
        <strong>{{ Number(mostFrequentGap?.gap ?? 0) }}</strong>
      </article>
      <article class="metric-card">
        <span>Hits at that gap</span>
        <strong>{{ Number(mostFrequentGap?.hits ?? 0).toLocaleString() }}</strong>
      </article>
    </div>

    <aside class="warning-banner">
      An opportunity is one number entering one draw at that gap. For fair,
      independent 6-from-49 draws, each opportunity has a 6/49 (about 12.24%)
      chance of a hit, regardless of gap. Expected hits equal the observed
      opportunities × 6/49. Positive differences mean more hits than this
      baseline; they do not imply a predictive advantage. Rate differences are
      in percentage points. Hover for opportunity counts, especially at sparse gaps.
    </aside>

    <article class="chart-card wide">
      <PlotlyChart :figure="figures.freshness_gap_distribution" />
    </article>

    <article class="chart-card wide">
      <PlotlyChart :figure="figures.freshness_gap_hit_rate" />
    </article>

    <article class="table-card">
      <h3>Exact gap statistics</h3>
      <DataTable
        :table="analysis.tables.freshness_gap_distribution"
        :searchable="false"
      />
    </article>
  </section>
</template>
