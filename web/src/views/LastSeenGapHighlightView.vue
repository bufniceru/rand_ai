<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import { buildLastSeenGapModel } from "../lib/lastSeenGap";
import type { HistoryDraw } from "../types";

const props = defineProps<{
  history: HistoryDraw[];
  drawCount: number;
  referenceDrawOffset: number;
}>();

const baseSvgWidth = 1680;
const chartLeft = 70;
const chartTop = 30;
const chartBottom = 45;
const chartRight = 30;
const rowHeight = 30;
const pointRadius = 13.5;
const occurrenceStripWidth = (pointRadius * 2) / 3;
const basePlotWidth = baseSvgWidth - chartLeft - chartRight;
const horizontalUnitSpacing = basePlotWidth / 48;

const selectedGap = ref<number | null>(null);
const gapNumbersPopup = ref<{ gap: number; numbers: number[] } | null>(null);
let longPressTimer: ReturnType<typeof setTimeout> | null = null;

const model = computed(() =>
  buildLastSeenGapModel(
    props.history,
    props.drawCount,
    props.referenceDrawOffset,
  ),
);
const chartPlotWidth = computed(() =>
  Math.max(basePlotWidth, model.value.maxGap * horizontalUnitSpacing),
);
const svgWidth = computed(() => chartLeft + chartPlotWidth.value + chartRight);
const chartHeight = computed(() =>
  Math.max(320, chartTop + model.value.drawCount * rowHeight + chartBottom),
);
const plotHeight = computed(() => Math.max(1, model.value.drawCount - 1) * rowHeight);
const rowDrawIndices = computed(() =>
  Array.from(
    { length: model.value.drawCount },
    (_value, index) => model.value.drawCount - 1 - index,
  ),
);
const referenceDrawGaps = computed(() => new Set(model.value.referenceGaps));
const gapUnits = computed(() =>
  Array.from({ length: model.value.maxGap + 1 }, (_value, gap) => gap),
);
const gapTicks = computed(() => {
  const step = model.value.maxGap > 60 ? 5 : 1;
  return gapUnits.value.filter((gap) => gap % step === 0);
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
const drawGapBands = computed(() => {
  const ranges = new Map<number, { first: number; last: number }>();
  for (const point of model.value.points) {
    const range = ranges.get(point.drawIndex);
    if (range) {
      range.first = Math.min(range.first, point.gap);
      range.last = Math.max(range.last, point.gap);
    } else {
      ranges.set(point.drawIndex, {
        first: point.gap,
        last: point.gap,
      });
    }
  }
  return [...ranges.entries()]
    .filter(([_drawIndex, range]) => range.first < range.last)
    .map(([drawIndex, range]) => ({ drawIndex, ...range }));
});
const occurrenceStrips = computed(() => {
  const referenceIndex = model.value.referenceDrawIndex;
  if (referenceIndex === null) return [];

  // A draw can contain the same gap more than once. Keep one strip per gap so
  // overlapping SVG rectangles cannot produce inconsistent opacity.
  const strips = new Map<
    number,
    { gap: number; x: number; y: number; height: number }
  >();
  for (const point of model.value.points) {
    if (!point.highlighted || point.drawIndex >= referenceIndex) continue;
    const topY = yForDraw(referenceIndex) + pointRadius;
    const bottomY = yForDraw(point.drawIndex) - pointRadius;
    const height = Math.max(0, bottomY - topY);
    if (height > 0) {
      strips.set(point.gap, {
        gap: point.gap,
        x: xForGap(point.gap) - occurrenceStripWidth / 2,
        y: topY,
        height,
      });
    }
  }
  return [...strips.values()];
});
const referencePrecedentStrips = computed(() => {
  const referenceIndex = model.value.referenceDrawIndex;
  if (referenceIndex === null) return [];
  const referenceGaps = new Set(
    model.value.points
      .filter((point) => point.drawIndex === referenceIndex)
      .map((point) => point.gap),
  );
  return [...referenceGaps]
    .map((gap) => {
      const precedent = model.value.points
        .filter((point) => point.gap === gap && point.drawIndex < referenceIndex)
        .sort((left, right) => right.drawIndex - left.drawIndex)[0];
      if (!precedent) return null;
      const topY = yForDraw(referenceIndex) + pointRadius;
      const bottomY = yForDraw(precedent.drawIndex) - pointRadius;
      return {
        gap,
        x: xForGap(gap) - occurrenceStripWidth / 2,
        y: topY,
        height: Math.max(0, bottomY - topY),
      };
    })
    .filter(
      (strip): strip is {
        gap: number;
        x: number;
        y: number;
        height: number;
      } => strip !== null && strip.height > 0,
    );
});

function xForGap(gap: number): number {
  return chartLeft + gap * horizontalUnitSpacing;
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

function clearLongPressTimer(): void {
  if (longPressTimer !== null) {
    clearTimeout(longPressTimer);
    longPressTimer = null;
  }
}

function startLongPress(gap: number): void {
  clearLongPressTimer();
  longPressTimer = setTimeout(() => {
    selectedGap.value = gap;
    longPressTimer = null;
  }, 1000);
}

function openGapNumbersPopup(event: MouseEvent, gap: number): void {
  if (!event.ctrlKey) return;
  event.preventDefault();
  event.stopPropagation();
  clearLongPressTimer();
  gapNumbersPopup.value = {
    gap,
    numbers: model.value.referenceGapNumbers[gap] ?? [],
  };
}

onBeforeUnmount(clearLongPressTimer);
</script>

<template>
  <section class="workspace-view last-seen-view gap-highlight-view">
    <div class="highlight-chart-scroll">
      <svg :width="svgWidth" height="42" class="highlight-chart-header" role="presentation">
        <g
          v-for="gap in gapUnits"
          :key="`header-gap-${gap}`"
          class="gap-ribbon-label"
          @click="openGapNumbersPopup($event, gap)"
        >
          <circle
            :class="{ matched: referenceDrawGaps.has(gap) }"
            :cx="xForGap(gap)"
            cy="21"
            :r="pointRadius"
            class="top-number-circle"
          />
          <text
            :class="{ compact: gap >= 100 }"
            :x="xForGap(gap)"
            y="26"
            class="top-number-circle-label"
          >
            {{ gap }}
            <title>Ctrl+Click to show numbers with gap {{ gap }}</title>
          </text>
        </g>
      </svg>
      <svg :height="chartHeight" :width="svgWidth" class="highlight-chart" role="img">
        <text class="axis-label" :x="svgWidth / 2" :y="chartHeight - 10">Gap</text>
        <text
          class="axis-label"
          :x="18"
          :y="chartHeight / 2"
          transform="rotate(-90, 18, 240)"
        >Draw Index</text>

        <line
          v-for="gap in gapUnits"
          :key="`vertical-${gap}`"
          :class="{ major: gap % 5 === 0 }"
          class="vertical-guide gap-unit-guide"
          :x1="xForGap(gap)"
          :x2="xForGap(gap)"
          :y1="chartTop - 10"
          :y2="chartTop + plotHeight + 10"
        />

        <rect
          v-if="model.referenceDrawIndex !== null"
          :x="chartLeft - 11"
          :y="yForDraw(model.referenceDrawIndex) - 14"
          :width="chartPlotWidth + 22"
          height="28"
          class="current-reference-ribbon"
        />

        <g v-for="drawIndex in rowDrawIndices" :key="`horizontal-${drawIndex}`">
          <line
            :class="{ major: (model.drawCount - drawIndex) % 5 === 0 }"
            class="horizontal-guide"
            :x1="chartLeft"
            :x2="chartLeft + chartPlotWidth"
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
          v-for="gap in gapTicks"
          :key="`bottom-${gap}`"
          :class="{ major: gap % 5 === 0 }"
          :x="xForGap(gap)"
          :y="chartTop + plotHeight + 28"
          class="tick-label x-tick"
        >{{ gap }}</text>

        <rect
          v-for="band in drawGapBands"
          :key="`draw-gap-band-${band.drawIndex}`"
          :x="xForGap(band.first) - pointRadius"
          :y="yForDraw(band.drawIndex) - pointRadius"
          :width="xForGap(band.last) - xForGap(band.first) + pointRadius * 2"
          :height="pointRadius * 2"
          :rx="pointRadius"
          class="draw-occurrence-band"
        />

        <rect
          v-for="strip in occurrenceStrips"
          :key="`gap-interval-${strip.gap}`"
          :x="strip.x"
          :y="strip.y"
          :width="occurrenceStripWidth"
          :height="strip.height"
          class="occurrence-interval-strip"
          rx="4.5"
        />
        <rect
          v-for="strip in referencePrecedentStrips"
          :key="`gap-precedent-${strip.gap}`"
          :x="strip.x"
          :y="strip.y"
          :width="occurrenceStripWidth"
          :height="strip.height"
          class="reference-precedent-strip"
          rx="4.5"
        />

        <g
          v-for="(point, index) in model.points"
          :key="`${point.drawIndex}-${point.gap}-${index}`"
          class="highlight-point"
        >
          <circle
            :class="pointClass(point.drawIndex, point.highlighted)"
            :cx="xForGap(point.gap)"
            :cy="yForDraw(point.drawIndex)"
            :r="pointRadius"
            @pointercancel="clearLongPressTimer"
            @pointerdown="startLongPress(point.gap)"
            @pointerleave="clearLongPressTimer"
            @pointerup="clearLongPressTimer"
          />
          <text
            :x="xForGap(point.gap)"
            :y="yForDraw(point.drawIndex) + 5"
            class="point-label"
          >{{ point.gapGap }}</text>
          <title>Gap {{ point.gap }}, gap since previous occurrence {{ point.gapGap }}</title>
        </g>
      </svg>
    </div>

    <div
      v-if="selectedGap !== null"
      class="number-popup-backdrop"
      role="presentation"
      @click.self="selectedGap = null"
    >
      <section class="number-popup">
        <p>Selected gap</p>
        <strong>{{ selectedGap }}</strong>
        <button type="button" @click="selectedGap = null">Close</button>
      </section>
    </div>

    <div
      v-if="gapNumbersPopup"
      class="number-popup-backdrop"
      role="presentation"
      @click.self="gapNumbersPopup = null"
    >
      <section class="number-popup gap-numbers-popup">
        <p>Numbers with gap {{ gapNumbersPopup.gap }}</p>
        <div v-if="gapNumbersPopup.numbers.length" class="gap-number-list">
          <span
            v-for="number in gapNumbersPopup.numbers"
            :key="number"
            class="gap-number-ball"
          >{{ number }}</span>
        </div>
        <span v-else>No numbers have this gap.</span>
        <button type="button" @click="gapNumbersPopup = null">Close</button>
      </section>
    </div>
  </section>
</template>
