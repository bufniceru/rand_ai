<script setup lang="ts">
import { computed } from "vue";
import {
  buildEfficacyChartScale,
  efficacyBarGeometry,
  formatEfficacyChartValue,
  type EfficacyChartMode,
  type EfficacyChartRow,
} from "../lib/efficacyChart";

const props = withDefaults(
  defineProps<{
    id: string;
    title: string;
    rows: readonly EfficacyChartRow[];
    mode: EfficacyChartMode;
    rateUnit: string;
    randomRate?: number;
  }>(),
  { randomRate: 36 / 49 },
);

const scale = computed(() =>
  buildEfficacyChartScale(props.rows, props.mode, props.randomRate),
);
const titleId = computed(() => `${props.id}-title`);

function valueFor(row: EfficacyChartRow): number {
  return props.mode === "rate" ? row.rate : row.normalizedLift;
}

function barStyle(row: EfficacyChartRow): Record<string, string> {
  const geometry = efficacyBarGeometry(
    valueFor(row),
    props.mode,
    scale.value.maximum,
  );
  return {
    left: `${geometry.leftPercent}%`,
    width: `${geometry.widthPercent}%`,
  };
}

function resultClass(row: EfficacyChartRow): string {
  if (row.normalizedLift > 0) return "is-ahead";
  if (row.normalizedLift < 0) return "is-behind";
  return "is-tied";
}

function valueLabel(row: EfficacyChartRow): string {
  return formatEfficacyChartValue(valueFor(row), props.mode);
}

function accessibleLabel(row: EfficacyChartRow, index: number): string {
  const rate = `${row.rate.toFixed(3)} hits per ${props.rateUnit}`;
  const lift = formatEfficacyChartValue(row.normalizedLift, "lift");
  const comparison =
    row.normalizedLift > 0
      ? `${lift} above random`
      : row.normalizedLift < 0
        ? `${Math.abs(row.normalizedLift).toFixed(3)} below random`
        : "tied with random";
  return [
    `Rank ${index + 1}, ${row.label}: ${rate}, ${comparison}.`,
    row.detail,
  ]
    .filter(Boolean)
    .join(" ");
}
</script>

<template>
  <section class="efficacy-comparison-chart" :aria-labelledby="titleId">
    <header>
      <h3 :id="titleId">{{ title }}</h3>
      <small v-if="mode === 'rate'">
        Dashed line: random expectation {{ randomRate.toFixed(3) }}
      </small>
      <small v-else>Zero-centered lift from random expectation</small>
    </header>

    <ol v-if="rows.length > 0">
      <li
        v-for="(row, index) in rows"
        :key="row.id"
        :class="resultClass(row)"
        role="img"
        :aria-label="accessibleLabel(row, index)"
      >
        <span class="efficacy-chart-label" :title="row.detail || row.label">
          <b>#{{ index + 1 }}</b>
          {{ row.label }}
        </span>
        <span class="efficacy-chart-track" aria-hidden="true">
          <i
            class="efficacy-chart-reference"
            :style="{ left: `${scale.referencePercent}%` }"
          ></i>
          <i
            class="efficacy-chart-bar"
            :class="{ 'is-zero': valueFor(row) === 0 }"
            :style="barStyle(row)"
          ></i>
        </span>
        <strong>{{ valueLabel(row) }}</strong>
      </li>
    </ol>
    <p v-else>No efficacy results are available for this range.</p>
  </section>
</template>

<style scoped>
.efficacy-comparison-chart {
  min-width: 0;
  margin-top: 10px;
  padding-top: 9px;
  border-top: 1px solid var(--monokai-border, #c8d8e7);
  break-inside: avoid;
}

.efficacy-comparison-chart > header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 7px;
}

.efficacy-comparison-chart h3 {
  margin: 0;
  color: var(--monokai-fg, #173d64);
  font-size: 13px;
}

.efficacy-comparison-chart header small,
.efficacy-comparison-chart > p {
  color: var(--monokai-muted, #687d92);
  font-size: 10px;
  font-weight: 750;
}

.efficacy-comparison-chart ol {
  display: grid;
  gap: 3px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.efficacy-comparison-chart li {
  min-width: 0;
  display: grid;
  grid-template-columns: minmax(130px, 0.8fr) minmax(180px, 2fr) 56px;
  align-items: center;
  gap: 8px;
  min-height: 24px;
}

.efficacy-chart-label {
  min-width: 0;
  overflow: hidden;
  color: var(--monokai-muted, #496984);
  font-size: 10px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.efficacy-chart-label b {
  display: inline-block;
  width: 25px;
  color: var(--monokai-cyan, #3377b2);
  font-size: 9px;
}

.efficacy-chart-track {
  position: relative;
  height: 15px;
  overflow: hidden;
  border: 1px solid var(--monokai-border, #c8d8e7);
  background: color-mix(
    in srgb,
    var(--monokai-deep, #eef4f9) 82%,
    transparent
  );
}

.efficacy-chart-reference {
  position: absolute;
  z-index: 2;
  top: -1px;
  bottom: -1px;
  width: 0;
  border-left: 1px dashed var(--monokai-yellow, #9b6e00);
  pointer-events: none;
}

.efficacy-chart-bar {
  position: absolute;
  z-index: 1;
  top: 2px;
  bottom: 2px;
  min-width: 1px;
  background: var(--monokai-cyan, #3377b2);
}

.is-ahead .efficacy-chart-bar {
  background: var(--monokai-green, #2d7b3f);
}

.is-behind .efficacy-chart-bar {
  background: var(--monokai-pink, #bd2a5a);
}

.is-tied .efficacy-chart-bar {
  background: var(--monokai-cyan, #3377b2);
}

.efficacy-chart-bar.is-zero {
  width: 2px !important;
  transform: translateX(-1px);
}

.efficacy-comparison-chart li > strong {
  color: var(--monokai-fg, #173d64);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  text-align: right;
}

@media (max-width: 760px) {
  .efficacy-comparison-chart > header {
    align-items: flex-start;
    flex-direction: column;
    gap: 2px;
  }

  .efficacy-comparison-chart li {
    grid-template-columns: minmax(100px, 0.8fr) minmax(120px, 1.5fr) 48px;
    gap: 5px;
  }
}

@media print {
  .efficacy-comparison-chart {
    border-color: #bccbd8 !important;
  }

  .efficacy-comparison-chart h3,
  .efficacy-comparison-chart li > strong {
    color: #173d64 !important;
  }

  .efficacy-chart-track {
    border-color: #bccbd8 !important;
    background: #f4f7fa !important;
  }

  .efficacy-chart-reference {
    border-color: #886100 !important;
  }

  .is-ahead .efficacy-chart-bar {
    background: #2d7b3f !important;
  }

  .is-behind .efficacy-chart-bar {
    background: #bd2a5a !important;
  }
}
</style>
