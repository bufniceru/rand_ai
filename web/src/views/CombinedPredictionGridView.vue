<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type {
  CombinedPredictionHistory,
  CombinedPredictionNumber,
} from "../types";

const props = defineProps<{
  predictions: CombinedPredictionHistory[];
}>();

const referenceOffset = ref(0);

const maximumOffset = computed(() => Math.max(0, props.predictions.length - 1));
const selectedIndex = computed(() =>
  Math.max(0, props.predictions.length - 1 - referenceOffset.value),
);
const selectedPrediction = computed(() => props.predictions[selectedIndex.value] ?? null);
const actualNumbers = computed(
  () => new Set(selectedPrediction.value?.actualNumbers ?? []),
);

watch(
  () => props.predictions.length,
  () => {
    referenceOffset.value = Math.min(referenceOffset.value, maximumOffset.value);
  },
);

function scoreLabel(entry: CombinedPredictionNumber): string {
  return `${(entry.score * 100).toFixed(2)}%`;
}

function cellTitle(entry: CombinedPredictionNumber): string {
  const outcome = actualNumbers.value.has(entry.number) ? " — drawn next" : "";
  return `Number ${entry.number}: rank ${entry.rank}, combined score ${scoreLabel(entry)}${outcome}`;
}
</script>

<template>
  <section v-if="selectedPrediction" class="combined-prediction-view">
    <header class="prediction-reference-toolbar">
      <div class="reference-buttons" aria-label="Prediction history navigation">
        <button
          type="button"
          :disabled="referenceOffset >= maximumOffset"
          @click="referenceOffset = maximumOffset"
        >
          First
        </button>
        <button
          type="button"
          :disabled="referenceOffset >= maximumOffset"
          @click="referenceOffset += 1"
        >
          Previous
        </button>
        <button
          type="button"
          :disabled="referenceOffset === 0"
          @click="referenceOffset -= 1"
        >
          Next
        </button>
        <button
          type="button"
          :disabled="referenceOffset === 0"
          @click="referenceOffset = 0"
        >
          Latest
        </button>
      </div>
      <div class="prediction-reference-summary">
        <span>Prediction after draw</span>
        <strong>{{ selectedPrediction.referenceDrawNumber }}</strong>
        <span aria-hidden="true">→</span>
        <span>draw {{ selectedPrediction.targetDrawNumber }}</span>
      </div>
    </header>

    <div class="prediction-outcome-note">
      <span v-if="selectedPrediction.actualNumbers.length > 0">
        ✓ marks a number drawn in draw {{ selectedPrediction.targetDrawNumber }}
      </span>
      <small v-if="selectedPrediction.actualNumbers.length === 0">
        The next draw is not recorded yet.
      </small>
    </div>

    <div class="combined-score-grid" role="grid" aria-label="Numbers 1 through 49">
      <div
        v-for="entry in selectedPrediction.numbers"
        :key="entry.number"
        class="combined-score-cell"
        :class="{ 'is-drawn': actualNumbers.has(entry.number) }"
        :title="cellTitle(entry)"
        role="gridcell"
      >
        <span class="prediction-rank">#{{ entry.rank }}</span>
        <strong>{{ entry.number }}</strong>
        <small>{{ scoreLabel(entry) }}</small>
        <span
          v-if="actualNumbers.has(entry.number)"
          class="actual-marker"
          aria-label="Drawn next"
        >
          ✓
        </span>
      </div>
    </div>
  </section>

  <section v-else class="dialog-empty-state">
    <strong>No predictions are available</strong>
    <p>The imported dataset does not contain any draws.</p>
  </section>
</template>
