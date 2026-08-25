<script setup lang="ts">
import { computed } from "vue";

import DataTable from "../components/DataTable.vue";
import PlotlyChart from "../components/PlotlyChart.vue";
import type { AnalysisPayload, FigureSpec } from "../types";

const props = defineProps<{
  analysis: AnalysisPayload;
  figures: Record<string, FigureSpec>;
}>();

const model = computed(() => props.analysis.spaceGroups);
const bestName = computed(() =>
  model.value?.forecasts.find((forecast) => forecast.modelId === model.value?.bestModelId)?.name
    ?? "Hybrid fallback",
);
</script>

<template>
  <section class="workspace-view border-groups-view">
    <header class="view-header">
      <div>
        <p class="eyebrow">Circular structure</p>
        <h2>Border Groups</h2>
      </div>
      <p v-if="model">
        Spaces {{ model.smallSpaceDefinition }} connect numbers; spaces
        {{ model.largeSpaceDefinition }} start a new group.
      </p>
    </header>

    <section v-if="model" class="border-group-summary">
      <article>
        <span>Border</span>
        <strong>{{ model.borderSpace }}</strong>
        <small>
          Inclusive limit · Target:
          {{ model.targetGroupCount === null ? "Automatic" : `${model.targetGroupCount} groups` }}
        </small>
      </article>
      <article>
        <span>Recommended model</span>
        <strong>{{ bestName }}</strong>
        <small>{{ model.provisional ? "Provisional: fewer than 100 targets" : "Selected by walk-forward log loss" }}</small>
      </article>
      <article>
        <span>Signature vs random</span>
        <strong>p={{ model.signatureChiSquarePValue.toFixed(4) }}</strong>
        <small>Exact 6/49 null comparison</small>
      </article>
      <article>
        <span>Temporal dependence</span>
        <strong>p={{ model.transitionPermutationPValue.toFixed(4) }}</strong>
        <small>Seeded transition permutation test</small>
      </article>
    </section>

    <section v-if="model" class="border-group-forecast-grid">
      <article
        v-for="forecast in model.forecasts"
        :key="forecast.modelId"
        :class="{ recommended: forecast.modelId === model.bestModelId }"
      >
        <span>{{ forecast.name }}</span>
        <strong>{{ forecast.topSignature }}</strong>
        <small>
          {{ forecast.topGroupCount }} groups ·
          {{ (forecast.topProbability * 100).toFixed(1) }}%
        </small>
        <div class="border-group-probabilities">
          <i
            v-for="probability in forecast.probabilities.slice().sort((left, right) => right.probability - left.probability).slice(0, 3)"
            :key="probability.signature"
          >
            {{ probability.signature }} {{ (probability.probability * 100).toFixed(1) }}%
          </i>
        </div>
      </article>
    </section>

    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.border_group_counts" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.border_group_signatures" /></article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.border_group_history" /></article>
    <div class="chart-grid">
      <article class="chart-card"><PlotlyChart :figure="figures.border_group_transitions" /></article>
      <article class="chart-card"><PlotlyChart :figure="figures.border_group_sensitivity" /></article>
    </div>
    <article class="chart-card wide"><PlotlyChart :figure="figures.border_group_models" /></article>

    <article class="table-card">
      <h3>Walk-forward model comparison</h3>
      <DataTable :table="analysis.tables.space_group_model_metrics" :searchable="false" />
    </article>
    <article class="table-card">
      <h3>Draw-by-draw border groups</h3>
      <DataTable :table="analysis.tables.space_group_history" />
    </article>
  </section>
</template>
