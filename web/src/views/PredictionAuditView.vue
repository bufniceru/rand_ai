<script setup lang="ts">
import { computed, ref } from "vue";
import type {
  AnalysisPayload,
  PredictionAuditNumber,
  PredictionAuditRecord,
  StrategyId,
} from "../types";

const props = defineProps<{ analysis: AnalysisPayload }>();

type HistoryScope = "all" | 100 | 250 | 500;

interface SelectedOccurrence {
  record: PredictionAuditRecord;
  item: PredictionAuditNumber;
}

const selectedStrategyId = ref<"all" | StrategyId>("all");
const historyScope = ref<HistoryScope>("all");
const selectedOccurrence = ref<SelectedOccurrence | null>(null);

const records = computed(() => props.analysis.predictionAuditHistory);
const strategyCatalog = computed(() => {
  const catalog = new Map<StrategyId, string>();
  for (const record of records.value) {
    for (const item of record.numbers) {
      for (const strategy of item.strategies) {
        catalog.set(strategy.id, strategy.name);
      }
    }
  }
  return [...catalog].map(([id, name]) => ({ id, name }));
});
const visibleRecords = computed(() => {
  if (historyScope.value === "all") return records.value;
  return records.value.slice(-historyScope.value);
});
const occurrences = computed(() =>
  records.value.flatMap((record) =>
    record.numbers.map((item) => ({ record, item })),
  ),
);
const filteredStrategyName = computed(() =>
  selectedStrategyId.value === "all"
    ? "all active strategies"
    : strategyCatalog.value.find(
        (strategy) => strategy.id === selectedStrategyId.value,
      )?.name ?? selectedStrategyId.value,
);
const totalCorrectImplications = computed(() =>
  occurrences.value.reduce(
    (total, occurrence) =>
      total + filteredStrategies(occurrence.item).length,
    0,
  ),
);
const coveredOccurrences = computed(
  () =>
    occurrences.value.filter(
      (occurrence) => filteredStrategies(occurrence.item).length > 0,
    ).length,
);
const coverageRate = computed(() =>
  occurrences.value.length > 0
    ? coveredOccurrences.value / occurrences.value.length
    : 0,
);
const numberSummary = computed(() =>
  Array.from({ length: 49 }, (_value, index) => {
    const number = index + 1;
    const numberOccurrences = occurrences.value.filter(
      (occurrence) => occurrence.item.number === number,
    );
    const strategyCounts = new Map<string, { name: string; count: number }>();
    let predictedOccurrences = 0;
    let strategyHits = 0;
    for (const occurrence of numberOccurrences) {
      const strategies = filteredStrategies(occurrence.item);
      if (strategies.length > 0) predictedOccurrences += 1;
      strategyHits += strategies.length;
      for (const strategy of strategies) {
        const current = strategyCounts.get(strategy.id);
        strategyCounts.set(strategy.id, {
          name: strategy.name,
          count: (current?.count ?? 0) + 1,
        });
      }
    }
    const leaders = [...strategyCounts.values()]
      .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name))
      .slice(0, 3);
    return {
      number,
      drawn: numberOccurrences.length,
      predictedOccurrences,
      strategyHits,
      coverage:
        numberOccurrences.length > 0
          ? predictedOccurrences / numberOccurrences.length
          : 0,
      leaders,
    };
  }),
);

const chartLeft = 46;
const chartTop = 22;
const chartBottom = 52;
const chartRight = 28;
const rowHeight = 12;
const chartHeight = chartTop + 49 * rowHeight + chartBottom;
const columnWidth = computed(() =>
  Math.max(
    visibleRecords.value.length > 1200
      ? 5
      : visibleRecords.value.length > 600
        ? 8
        : 13,
    visibleRecords.value.length > 1
      ? (940 - chartLeft - chartRight) / (visibleRecords.value.length - 1)
      : 13,
  ),
);
const chartWidth = computed(() =>
  Math.max(
    940,
    chartLeft + chartRight + Math.max(visibleRecords.value.length - 1, 1) * columnWidth.value,
  ),
);
const timelineTicks = computed(() => {
  const count = visibleRecords.value.length;
  if (count === 0) return [];
  const tickCount = Math.min(10, count);
  const indexes = new Set<number>();
  for (let tick = 0; tick < tickCount; tick += 1) {
    indexes.add(Math.round((tick * (count - 1)) / Math.max(tickCount - 1, 1)));
  }
  return [...indexes].map((index) => ({
    index,
    draw: visibleRecords.value[index].targetDrawNumber,
  }));
});

function filteredStrategies(item: PredictionAuditNumber) {
  return selectedStrategyId.value === "all"
    ? item.strategies
    : item.strategies.filter(
        (strategy) => strategy.id === selectedStrategyId.value,
      );
}

function xForRecord(index: number): number {
  return chartLeft + index * columnWidth.value;
}

function yForNumber(number: number): number {
  return chartTop + (number - 0.5) * rowHeight;
}

function implicationColor(item: PredictionAuditNumber): string {
  const count = filteredStrategies(item).length;
  if (selectedStrategyId.value !== "all") return count > 0 ? "#a9dc76" : "#49464a";
  if (count === 0) return "#49464a";
  if (count <= 2) return "#78dce8";
  if (count <= 5) return "#a9dc76";
  if (count <= 8) return "#ab9df2";
  return "#ff6188";
}

function occurrenceTitle(
  record: PredictionAuditRecord,
  item: PredictionAuditNumber,
): string {
  const strategies = filteredStrategies(item);
  const names = strategies.map((strategy) => strategy.name).join(", ");
  return `Draw ${record.targetDrawNumber} · number ${item.number} · ${strategies.length} correct ${strategies.length === 1 ? "strategy" : "strategies"}${names ? `: ${names}` : ""}`;
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
</script>

<template>
  <section class="workspace-view prediction-audit-view">
    <header class="view-heading prediction-analysis-heading">
      <div>
        <p class="eyebrow">Walk-forward prediction history</p>
        <h1>Prediction audit</h1>
        <p>
          Every dot is an actual drawn number. Its color records how many
          active strategies had included that number in their prior Top 6.
        </p>
      </div>
      <div class="prediction-analysis-controls">
        <label>
          <span>Strategy</span>
          <select v-model="selectedStrategyId">
            <option value="all">All active strategies</option>
            <option
              v-for="strategy in strategyCatalog"
              :key="strategy.id"
              :value="strategy.id"
            >
              {{ strategy.name }}
            </option>
          </select>
        </label>
        <label>
          <span>History</span>
          <select v-model="historyScope">
            <option value="all">All evaluated draws</option>
            <option :value="100">Latest 100</option>
            <option :value="250">Latest 250</option>
            <option :value="500">Latest 500</option>
          </select>
        </label>
      </div>
    </header>

    <div class="prediction-analysis-facts">
      <article>
        <span>Evaluated draws</span>
        <strong>{{ records.length.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Drawn-number events</span>
        <strong>{{ occurrences.length.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Events predicted</span>
        <strong>{{ coveredOccurrences.toLocaleString() }}</strong>
        <small>{{ percentage(coverageRate) }} coverage</small>
      </article>
      <article>
        <span>Correct implications</span>
        <strong>{{ totalCorrectImplications.toLocaleString() }}</strong>
        <small>{{ filteredStrategyName }}</small>
      </article>
    </div>

    <section class="prediction-history-panel">
      <header>
        <div>
          <h2>Draw timeline × number</h2>
          <p>Click any drawn-number dot to inspect its successful strategies.</p>
        </div>
        <div class="audit-color-legend">
          <span><i style="--audit-color: #49464a" />0</span>
          <span><i style="--audit-color: #78dce8" />1–2</span>
          <span><i style="--audit-color: #a9dc76" />3–5</span>
          <span><i style="--audit-color: #ab9df2" />6–8</span>
          <span><i style="--audit-color: #ff6188" />9+</span>
        </div>
      </header>
      <div v-if="visibleRecords.length > 0" class="prediction-audit-chart-scroll">
        <svg
          :height="chartHeight"
          :viewBox="`0 0 ${chartWidth} ${chartHeight}`"
          :width="chartWidth"
          role="img"
          aria-label="Timeline of actual numbers and the strategies that predicted them"
        >
          <g v-for="number in 49" :key="`row-${number}`">
            <line
              v-if="number % 5 === 0"
              :x1="chartLeft"
              :x2="chartWidth - chartRight"
              :y1="yForNumber(number)"
              :y2="yForNumber(number)"
              class="audit-major-line"
            />
            <text
              :x="chartLeft - 12"
              :y="yForNumber(number) + 3"
              class="audit-number-label"
            >{{ number }}</text>
          </g>
          <g
            v-for="tick in timelineTicks"
            :key="`tick-${tick.draw}`"
          >
            <line
              :x1="xForRecord(tick.index)"
              :x2="xForRecord(tick.index)"
              :y1="chartTop"
              :y2="chartHeight - chartBottom + 4"
              class="audit-draw-line"
            />
            <text
              :x="xForRecord(tick.index)"
              :y="chartHeight - 22"
              class="audit-draw-label"
            >{{ tick.draw }}</text>
          </g>
          <g
            v-for="(record, recordIndex) in visibleRecords"
            :key="record.targetDrawNumber"
          >
            <circle
              v-for="item in record.numbers"
              :key="`${record.targetDrawNumber}-${item.number}`"
              :cx="xForRecord(recordIndex)"
              :cy="yForNumber(item.number)"
              :fill="implicationColor(item)"
              :aria-label="occurrenceTitle(record, item)"
              class="audit-occurrence"
              r="4.25"
              role="button"
              tabindex="0"
              @click="selectedOccurrence = { record, item }"
              @keydown.enter.prevent="selectedOccurrence = { record, item }"
              @keydown.space.prevent="selectedOccurrence = { record, item }"
            >
              <title>{{ occurrenceTitle(record, item) }}</title>
            </circle>
          </g>
          <text
            :x="chartWidth / 2"
            :y="chartHeight - 5"
            class="audit-axis-title"
          >Target draw number →</text>
        </svg>
      </div>
      <p v-else class="prediction-analysis-empty">
        No completed walk-forward predictions are available.
      </p>
    </section>

    <section
      v-if="selectedOccurrence"
      class="prediction-audit-selection"
      aria-live="polite"
    >
      <div class="audit-selected-ball">{{ selectedOccurrence.item.number }}</div>
      <div>
        <span>Target draw {{ selectedOccurrence.record.targetDrawNumber }}</span>
        <strong>
          {{ filteredStrategies(selectedOccurrence.item).length }}
          successful
          {{
            filteredStrategies(selectedOccurrence.item).length === 1
              ? "strategy"
              : "strategies"
          }}
        </strong>
        <small>{{ selectedOccurrence.record.date || "Date unavailable" }}</small>
      </div>
      <ul v-if="filteredStrategies(selectedOccurrence.item).length > 0">
        <li
          v-for="strategy in filteredStrategies(selectedOccurrence.item)"
          :key="strategy.id"
        >
          {{ strategy.name }}
        </li>
      </ul>
      <p v-else>No selected strategy predicted this drawn number.</p>
      <button type="button" @click="selectedOccurrence = null">Close detail</button>
    </section>

    <section class="prediction-number-summary">
      <header>
        <div>
          <h2>Per-number historical coverage</h2>
          <p>
            Draw frequency, correctly predicted occurrences, and the strategies
            most often responsible.
          </p>
        </div>
      </header>
      <div class="prediction-summary-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Number</th>
              <th>Times drawn</th>
              <th>Draws predicted</th>
              <th>Coverage</th>
              <th>Strategy hits</th>
              <th>Most frequent successful strategies</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in numberSummary" :key="row.number">
              <th scope="row"><span>{{ row.number }}</span></th>
              <td>{{ row.drawn }}</td>
              <td>{{ row.predictedOccurrences }}</td>
              <td>
                <div class="coverage-meter">
                  <i :style="{ width: percentage(row.coverage) }" />
                  <strong>{{ percentage(row.coverage) }}</strong>
                </div>
              </td>
              <td>{{ row.strategyHits }}</td>
              <td>
                {{
                  row.leaders.length > 0
                    ? row.leaders.map((leader) => `${leader.name} (${leader.count})`).join(" · ")
                    : "None"
                }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </section>
</template>
