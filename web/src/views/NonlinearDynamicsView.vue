<script setup lang="ts">
import { computed } from "vue";
import DataTable from "../components/DataTable.vue";
import {
  NONLINEAR_STATUS_LABELS,
  RECURRENCE_FORECAST_VERSION_LABEL,
  recurrencePointPercent,
} from "../lib/nonlinearDynamics";
import type { AnalysisPayload } from "../types";

const props = defineProps<{ analysis: AnalysisPayload }>();

const model = computed(() => props.analysis.nonlinearDynamics);
const plotSize = computed(() => Math.max(model.value?.plot.size ?? 0, 1));
const plotPoints = computed(() => model.value?.plot.points ?? []);

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
</script>

<template>
  <section v-if="model" class="workspace-view nonlinear-dynamics-view">
    <header class="view-header">
      <div>
        <p class="eyebrow">Nonlinear time-series analysis</p>
        <h2>Nonlinear Dynamics</h2>
      </div>
      <p>Delay embeddings and recurrence tests applied to chronological draw structure.</p>
    </header>

    <aside class="warning-banner">{{ model.caveat }}</aside>

    <section class="nonlinear-evidence-panel" :data-status="model.status">
      <div>
        <span class="nonlinear-status">{{ NONLINEAR_STATUS_LABELS[model.status] }}</span>
        <h3>{{ model.summary }}</h3>
        <p>
          The verdict requires both recurrence beyond order-shuffled surrogates and
          walk-forward performance above the fixed random expectation.
        </p>
      </div>
      <dl>
        <div><dt>Analyzed draws</dt><dd>{{ model.drawCount }}</dd></div>
        <div><dt>Embedded states</dt><dd>{{ model.embeddingCount }}</dd></div>
        <div><dt>Embedding depth</dt><dd>{{ model.embeddingDimension }} draws</dd></div>
        <div><dt>Surrogate p-value</dt><dd>{{ model.surrogate.pValue.toFixed(3) }}</dd></div>
      </dl>
    </section>

    <div class="nonlinear-layout">
      <article class="nonlinear-card recurrence-card">
        <div class="nonlinear-card-heading">
          <div><p class="eyebrow">Latest {{ model.plot.size }} states</p><h3>Recurrence plot</h3></div>
          <span>{{ percent(model.metrics.recurrenceRate) }} recurrence</span>
        </div>
        <svg
          class="recurrence-plot"
          viewBox="0 0 100 100"
          role="img"
          aria-label="Recurrence matrix of delay-embedded draw states"
        >
          <rect class="recurrence-background" x="0" y="0" width="100" height="100" />
          <circle
            v-for="(point, index) in plotPoints"
            :key="`${point.x}-${point.y}-${index}`"
            class="recurrence-point"
            :cx="recurrencePointPercent(point.x, plotSize)"
            :cy="recurrencePointPercent(point.y, plotSize)"
            :r="Math.max(0.12, 42 / plotSize)"
          />
        </svg>
        <p class="nonlinear-note">
          A point marks two embedded states closer than the fixed 10% recurrence threshold.
          The near-diagonal temporal neighborhood is excluded.
        </p>
      </article>

      <article class="nonlinear-card">
        <div class="nonlinear-card-heading"><div><p class="eyebrow">RQA</p><h3>Recurrence structure</h3></div></div>
        <dl class="nonlinear-metrics">
          <div><dt>Determinism</dt><dd>{{ percent(model.metrics.determinism) }}</dd></div>
          <div><dt>Surrogate mean</dt><dd>{{ percent(model.surrogate.meanDeterminism) }}</dd></div>
          <div><dt>Mean diagonal</dt><dd>{{ model.metrics.meanDiagonalLength.toFixed(2) }}</dd></div>
          <div><dt>Longest diagonal</dt><dd>{{ model.metrics.maximumDiagonalLength.toFixed(0) }}</dd></div>
          <div><dt>Laminarity</dt><dd>{{ percent(model.metrics.laminarity) }}</dd></div>
          <div><dt>Trapping time</dt><dd>{{ model.metrics.trappingTime.toFixed(2) }}</dd></div>
        </dl>
      </article>
    </div>

    <div class="nonlinear-layout">
      <article class="nonlinear-card">
        <div class="nonlinear-card-heading"><div><p class="eyebrow">{{ RECURRENCE_FORECAST_VERSION_LABEL }}</p><h3>Walk-forward forecast</h3></div></div>
        <dl class="nonlinear-metrics">
          <div><dt>Evaluated draws</dt><dd>{{ model.forecast.evaluatedDraws }}</dd></div>
          <div><dt>Mean hits/draw</dt><dd>{{ model.forecast.averageHitsPerDraw.toFixed(3) }}</dd></div>
          <div><dt>95% lower bound</dt><dd>{{ model.forecast.lowerConfidenceBound.toFixed(3) }}</dd></div>
          <div><dt>Random expectation</dt><dd>{{ model.forecast.expectedRandomHitsPerDraw.toFixed(3) }}</dd></div>
        </dl>
      </article>

      <article class="nonlinear-card">
        <div class="nonlinear-card-heading"><div><p class="eyebrow">Current neighborhood</p><h3>Latest analogue support</h3></div></div>
        <dl class="nonlinear-metrics">
          <div><dt>Analogues</dt><dd>{{ model.latest.analogueCount }}</dd></div>
          <div><dt>Effective neighbors</dt><dd>{{ model.latest.effectiveNeighbors.toFixed(1) }}</dd></div>
          <div><dt>Distance percentile</dt><dd>{{ percent(model.latest.distancePercentile) }}</dd></div>
          <div><dt>Evidence index</dt><dd>{{ percent(model.latest.evidenceScore) }}</dd></div>
        </dl>
        <div class="nonlinear-top-numbers" aria-label="Latest recurrence Top 6">
          <span v-for="number in model.latest.topNumbers" :key="number">{{ number }}</span>
        </div>
      </article>
    </div>

    <article class="table-card">
      <h3>Auditable metrics</h3>
      <DataTable :table="analysis.tables.nonlinear_dynamics_metrics" :searchable="false" />
    </article>
  </section>
</template>
