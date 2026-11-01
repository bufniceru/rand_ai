<script setup lang="ts">
import { computed } from "vue";
import type { AnalysisPayload } from "../types";
import { positionLabels as labels, useDirectionSelection } from "../lib/numberDirections";
const props = defineProps<{ analysis: AnalysisPayload }>();
const report = computed(() => props.analysis.firstNumberDirection);
const { selectedPosition, data } = useDirectionSelection(report);
const forecast = computed(() => data.value?.forecast);
const directions = ["lower", "equal", "higher"];
const percent = (value: number | null | undefined) => value == null ? "—" : `${(100 * value).toFixed(2)}%`;
const decimal = (value: number | null) => value == null ? "—" : value.toFixed(4);
</script>

<template>
  <section class="direction-view">
    <h2>Number Directions</h2>
    <p>Compare each sorted position in the next draw with the same position in the previous draw.</p>
    <p v-if="!report?.positions.some(item => item.forecast)">Load at least one draw to see a forecast.</p>
    <template v-else>
      <div class="direction-cards">
        <article v-for="item in report?.positions" :key="item.position" class="direction-card" :aria-label="`${labels[item.position - 1]} number forecast`">
          <h3>{{ labels[item.position - 1] }} number</h3>
          <template v-if="item.forecast">
            <p>Following draw {{ item.forecast.referenceDrawNumber }} · {{ item.forecast.referenceDate ?? "Undated" }}</p>
            <p>Previous value: <strong>{{ item.forecast.referenceValue }}</strong></p>
            <p>Next value: <strong>{{ item.forecast.predictedDirection }}</strong></p>
            <p>Active model: <strong>{{ item.forecast.activeModel }}</strong></p>
            <div v-for="(label, i) in directions" :key="label" class="direction-probability">
              <label :for="`direction-${item.position}-${label}`">{{ label }}: <strong>{{ percent(item.forecast.probabilities[i]) }}</strong> · Exact baseline {{ percent(item.forecast.baselineProbabilities[i]) }}</label>
              <progress :id="`direction-${item.position}-${label}`" :value="item.forecast.probabilities[i]" max="1" />
            </div>
            <p v-if="!item.forecast.activationEvidence.eligible">Learned-model activation pending: {{ item.forecast.activationEvidence.scoredCount }} completed forecasts at the last review; at least 200 required.</p>
          </template>
        </article>
      </div>
      <label class="direction-selector">Detailed report for
        <select v-model="selectedPosition" aria-label="Position for detailed report">
          <option v-for="(label, index) in labels" :key="index" :value="index + 1">{{ label }}</option>
        </select>
      </label>
      <h3>{{ labels[selectedPosition - 1] }} number details</h3>
      <p>Backtesting begins after 120 draws. Evidence is reviewed every 20 forecasts. Holm correction covers three candidates per position; it provides no simultaneous guarantee across six positions.</p>
      <h3>Chronological evaluation</h3>
      <p>Historical results are exploratory. The exact baseline assumes independent, fair 6-from-49 draws.</p>
      <div class="direction-scroll"><table>
        <thead><tr><th>Model</th><th>Forecasts</th><th>Log loss ↓</th><th>Brier ↓</th><th>Accuracy</th><th>Recall lower / equal / higher</th></tr></thead>
        <tbody><tr v-for="(summary, name) in data?.evaluation" :key="name"><th>{{ name }}</th><td>{{ summary.count }}</td><td>{{ decimal(summary.logLoss) }}</td><td>{{ decimal(summary.brier) }}</td><td>{{ percent(summary.accuracy) }}</td><td>{{ summary.recall.map(percent).join(" / ") }}</td></tr></tbody>
      </table></div>
      <details><summary>Activation evidence</summary><div class="direction-scroll"><table>
        <thead><tr><th>Candidate</th><th>Log-loss improvement</th><th>95% lower bound</th><th>Holm p</th><th>Supported</th></tr></thead>
        <tbody><tr v-for="(item, name) in forecast?.activationEvidence.candidates" :key="name"><th>{{ name }}</th><td>{{ decimal(item.improvement) }}</td><td>{{ decimal(item.lower95) }}</td><td>{{ decimal(item.holmPValue) }}</td><td>{{ item.supported ? "Yes" : "No" }}</td></tr></tbody>
      </table></div></details>
      <details><summary>Five chronological blocks and confusion matrices</summary>
        <div v-for="(block, index) in data?.blocks" :key="index"><h4>Block {{ index + 1 }}</h4>
          <p v-for="(item, name) in block" :key="name">{{ name }} · {{ item.count }} forecasts · loss {{ decimal(item.logLoss) }} · accuracy {{ percent(item.accuracy) }}</p>
        </div>
        <p>Confusion matrix rows are actual lower/equal/higher; columns are predicted lower/equal/higher.</p>
        <p v-for="(item, name) in data?.evaluation" :key="name">{{ name }}: {{ item.confusionMatrix }}</p>
      </details>
      <details><summary>Historical forecasts ({{ data?.history.length ?? 0 }})</summary><div class="direction-scroll"><table>
        <thead><tr><th>Target</th><th>Date</th><th>Previous value</th><th>Actual value</th><th>Lower / Equal / Higher</th><th>Predicted</th><th>Actual</th><th>Model</th></tr></thead>
        <tbody><tr v-for="row in data?.history" :key="row.targetDrawNumber"><td>{{ row.targetDrawNumber }}</td><td>{{ row.actualDate ?? "Undated" }}</td><td>{{ row.referenceValue }}</td><td>{{ row.actualValue }}</td><td>{{ row.probabilities.map(percent).join(" / ") }}</td><td>{{ row.predictedDirection }}</td><td>{{ row.actualDirection }}</td><td>{{ row.activeModel }}</td></tr></tbody>
      </table></div></details>
    </template>
  </section>
</template>

<style scoped>
.direction-view { padding: 24px; display: grid; gap: 16px; }
.direction-view h2, .direction-view h3, .direction-view p { margin: 0; }
.direction-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 16px; }
.direction-selector { display: flex; align-items: center; gap: 12px; }
.direction-selector select { padding: 8px; }
.direction-card { border: 1px solid currentColor; border-radius: 10px; padding: 20px; display: grid; gap: 16px; }
.direction-probability { display: grid; gap: 6px; }
progress { width: 100%; height: 18px; accent-color: var(--accent, #4285d4); }
.direction-scroll { overflow: auto; max-height: 560px; }
table { width: 100%; border-collapse: collapse; text-align: left; }
th, td { padding: 10px; border-bottom: 1px solid #8885; white-space: nowrap; }
summary { cursor: pointer; padding: 10px 0; }
</style>
