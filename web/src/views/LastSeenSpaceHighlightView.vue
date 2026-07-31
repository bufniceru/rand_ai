<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { buildLastSeenGapModel } from "../lib/lastSeenGap";
import { buildLastSeenSpaceModel } from "../lib/lastSeenSpace";
import type { HistoryDraw } from "../types";

const props = defineProps<{ history: HistoryDraw[] }>();

const svgWidth = 1680;
const chartLeft = 70;
const chartTop = 90;
const chartBottom = 45;
const chartRight = 30;
const rowHeight = 30;
const pointRadius = 13.5;
const plotWidth = svgWidth - chartLeft - chartRight;

const drawCount = ref(Math.min(50, props.history.length));
const referenceDrawOffset = ref(0);
const selectedSpace = ref<number | null>(null);
let longPressTimer: ReturnType<typeof setTimeout> | null = null;

watch(
  () => props.history,
  (history) => {
    drawCount.value = Math.min(Math.max(drawCount.value, 1), history.length);
    referenceDrawOffset.value = Math.min(
      referenceDrawOffset.value,
      Math.max(0, drawCount.value - 1),
    );
  },
);

const model = computed(() =>
  buildLastSeenSpaceModel(
    props.history,
    drawCount.value,
    referenceDrawOffset.value,
  ),
);
const gapScaleModel = computed(() =>
  buildLastSeenGapModel(
    props.history,
    drawCount.value,
    referenceDrawOffset.value,
  ),
);
const horizontalAxisMax = computed(() =>
  Math.max(model.value.maxSpace, gapScaleModel.value.maxGap),
);
const chartHeight = computed(() =>
  Math.max(320, chartTop + model.value.drawCount * rowHeight + chartBottom),
);
const plotHeight = computed(
  () => Math.max(1, model.value.drawCount - 1) * rowHeight,
);
const rowDrawIndices = computed(() =>
  Array.from(
    { length: model.value.drawCount },
    (_value, index) => model.value.drawCount - 1 - index,
  ),
);
const referenceDisplayIndex = computed(() =>
  model.value.referenceDrawIndex === null
    ? 0
    : model.value.drawCount - model.value.referenceDrawIndex,
);
const spaceUnits = computed(() =>
  Array.from(
    { length: horizontalAxisMax.value + 1 },
    (_value, space) => space,
  ),
);
const spaceTicks = computed(() => {
  const step = horizontalAxisMax.value > 60 ? 5 : 1;
  return spaceUnits.value.filter((space) => space % step === 0);
});
const yTicks = computed(() => {
  const step = Math.max(1, Math.floor(model.value.drawCount / 30));
  return rowDrawIndices.value
    .filter((_drawIndex, index) => index % step === 0)
    .map((drawIndex) => ({
      drawIndex,
      label: model.value.drawCount - drawIndex,
    }));
});

function xForSpace(space: number): number {
  if (horizontalAxisMax.value <= 0) return chartLeft + plotWidth / 2;
  return chartLeft + (space / horizontalAxisMax.value) * plotWidth;
}

function yForDraw(drawIndex: number): number {
  return chartTop + (model.value.drawCount - 1 - drawIndex) * rowHeight;
}

function pointClass(drawIndex: number, highlighted: boolean): string {
  if (
    model.value.referenceDrawIndex !== null &&
    drawIndex > model.value.referenceDrawIndex
  ) {
    return "point-reference-range";
  }
  return highlighted ? "point-highlighted" : "point-default";
}

function setDrawCount(event: Event): void {
  const value = Number((event.target as HTMLInputElement).value);
  drawCount.value = Math.min(
    Math.max(Math.trunc(value || 1), 1),
    props.history.length,
  );
  referenceDrawOffset.value = Math.min(
    referenceDrawOffset.value,
    drawCount.value - 1,
  );
}

function clearLongPressTimer(): void {
  if (longPressTimer !== null) {
    clearTimeout(longPressTimer);
    longPressTimer = null;
  }
}

function startLongPress(space: number): void {
  clearLongPressTimer();
  longPressTimer = setTimeout(() => {
    selectedSpace.value = space;
    longPressTimer = null;
  }, 1000);
}

onBeforeUnmount(clearLongPressTimer);
</script>

<template>
  <section class="workspace-view last-seen-view space-highlight-view">
    <section class="reference-toolbar">
      <label>
        <span>Draw count</span>
        <input
          :max="history.length"
          min="1"
          :value="drawCount"
          type="number"
          @change="setDrawCount"
        >
      </label>
      <div class="reference-buttons">
        <button
          :disabled="referenceDrawOffset === model.maxReferenceOffset"
          type="button"
          @click="referenceDrawOffset = model.maxReferenceOffset"
        >|&lt; First</button>
        <button
          :disabled="referenceDrawOffset === model.maxReferenceOffset"
          type="button"
          @click="referenceDrawOffset += 1"
        >&lt; Previous</button>
        <button
          :disabled="referenceDrawOffset === 0"
          type="button"
          @click="referenceDrawOffset -= 1"
        >Next &gt;</button>
        <button
          :disabled="referenceDrawOffset === 0"
          type="button"
          @click="referenceDrawOffset = 0"
        >Latest &gt;|</button>
      </div>
      <div class="reference-summary">
        <span>Reference history draw</span>
        <strong>{{ model.referenceDrawNumber?.toLocaleString() ?? "—" }}</strong>
        <small>Display index {{ referenceDisplayIndex }}</small>
      </div>
    </section>

    <aside class="space-conversion-note">
      <strong>Complete circular spaces:</strong>
      Rand AI internal space counts the unselected numbers between neighboring
      balls. All six spaces are shown, including the wraparound from the largest
      number through 49 and 1 to the smallest number.
    </aside>

    <div class="highlight-legend">
      <span><i class="legend-red" /> Last seen space</span>
      <span><i class="legend-blue" /> Earlier occurrence</span>
      <span><i class="legend-gray" /> Newer than reference</span>
      <span class="gap-help">Long-press a point to inspect its Rand AI space</span>
    </div>

    <div class="highlight-chart-scroll">
      <svg
        :height="chartHeight"
        :width="svgWidth"
        class="highlight-chart"
        role="img"
      >
        <text class="axis-label" :x="svgWidth / 2" :y="chartHeight - 10">
          Rand AI internal space
        </text>
        <text
          class="axis-label"
          :x="18"
          :y="chartHeight / 2"
          transform="rotate(-90, 18, 240)"
        >Draw Index</text>

        <line
          v-for="space in spaceUnits"
          :key="`vertical-${space}`"
          :class="{ major: space % 5 === 0 }"
          class="vertical-guide"
          :x1="xForSpace(space)"
          :x2="xForSpace(space)"
          :y1="chartTop - 10"
          :y2="chartTop + plotHeight + 10"
        />
        <rect
          v-if="model.referenceDrawIndex !== null"
          :x="xForSpace(0) - 11"
          :y="yForDraw(model.referenceDrawIndex) - 14"
          :width="plotWidth + 22"
          height="28"
          class="current-reference-ribbon"
        />
        <text
          v-for="space in spaceTicks.filter((value) => value % 5 === 0)"
          :key="`top-${space}`"
          :x="xForSpace(space)"
          :y="chartTop - 18"
          class="tick-label top-x-tick"
        >{{ space }}</text>

        <g v-for="drawIndex in rowDrawIndices" :key="`row-${drawIndex}`">
          <line
            :class="{ major: (model.drawCount - drawIndex) % 5 === 0 }"
            class="horizontal-guide"
            :x1="xForSpace(0)"
            :x2="xForSpace(horizontalAxisMax)"
            :y1="yForDraw(drawIndex)"
            :y2="yForDraw(drawIndex)"
          />
        </g>
        <text
          v-for="tick in yTicks"
          :key="`y-${tick.drawIndex}`"
          :x="chartLeft - 16"
          :y="yForDraw(tick.drawIndex) + 4"
          class="tick-label y-tick"
        >{{ tick.label }}</text>
        <text
          v-for="space in spaceTicks"
          :key="`bottom-${space}`"
          :class="{ major: space % 5 === 0 }"
          :x="xForSpace(space)"
          :y="chartTop + plotHeight + 28"
          class="tick-label x-tick"
        >{{ space }}</text>

        <g
          v-for="(point, index) in model.points"
          :key="`${point.drawIndex}-${point.space}-${index}`"
          class="highlight-point"
        >
          <circle
            :class="pointClass(point.drawIndex, point.highlighted)"
            :cx="xForSpace(point.space)"
            :cy="yForDraw(point.drawIndex)"
            :r="pointRadius"
            @pointercancel="clearLongPressTimer"
            @pointerdown="startLongPress(point.space)"
            @pointerleave="clearLongPressTimer"
            @pointerup="clearLongPressTimer"
          />
          <text
            :x="xForSpace(point.space)"
            :y="yForDraw(point.drawIndex) + 5"
            class="point-label"
            @pointercancel="clearLongPressTimer"
            @pointerdown="startLongPress(point.space)"
            @pointerleave="clearLongPressTimer"
            @pointerup="clearLongPressTimer"
          >{{ point.space }}</text>
          <title>
            Rand AI space {{ point.space }} · PyLotto difference
            {{ point.space + 1 }}
          </title>
        </g>
      </svg>
    </div>

    <div
      v-if="selectedSpace !== null"
      class="number-popup-backdrop"
      role="presentation"
      @click.self="selectedSpace = null"
    >
      <section class="number-popup">
        <p>Selected Rand AI space</p>
        <strong>{{ selectedSpace }}</strong>
        <span>PyLotto difference {{ selectedSpace + 1 }}</span>
        <button type="button" @click="selectedSpace = null">Close</button>
      </section>
    </div>
  </section>
</template>
