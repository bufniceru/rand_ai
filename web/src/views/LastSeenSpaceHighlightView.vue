<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import { buildLastSeenGapModel } from "../lib/lastSeenGap";
import { buildLastSeenSpaceModel } from "../lib/lastSeenSpace";
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

const selectedSpace = ref<number | null>(null);
let longPressTimer: ReturnType<typeof setTimeout> | null = null;

const model = computed(() =>
  buildLastSeenSpaceModel(
    props.history,
    props.drawCount,
    props.referenceDrawOffset,
  ),
);
const gapScaleModel = computed(() =>
  buildLastSeenGapModel(
    props.history,
    props.drawCount,
    props.referenceDrawOffset,
  ),
);
const horizontalAxisMax = computed(() =>
  Math.max(model.value.maxSpace, gapScaleModel.value.maxGap),
);
const chartPlotWidth = computed(() =>
  Math.max(basePlotWidth, horizontalAxisMax.value * horizontalUnitSpacing),
);
const svgWidth = computed(() => chartLeft + chartPlotWidth.value + chartRight);
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
const drawSpaceBands = computed(() => {
  const ranges = new Map<number, { first: number; last: number }>();
  for (const point of model.value.points) {
    const range = ranges.get(point.drawIndex);
    if (range) {
      range.first = Math.min(range.first, point.space);
      range.last = Math.max(range.last, point.space);
    } else {
      ranges.set(point.drawIndex, {
        first: point.space,
        last: point.space,
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

  // Repeated equal spaces in one draw share one strip, keeping the opacity
  // uniform across every space column.
  const strips = new Map<
    number,
    { space: number; x: number; y: number; height: number }
  >();
  for (const point of model.value.points) {
    if (!point.highlighted || point.drawIndex >= referenceIndex) continue;
    const topY = yForDraw(referenceIndex) + pointRadius;
    const bottomY = yForDraw(point.drawIndex) - pointRadius;
    const height = Math.max(0, bottomY - topY);
    if (height > 0) {
      strips.set(point.space, {
        space: point.space,
        x: xForSpace(point.space) - occurrenceStripWidth / 2,
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
  const referenceSpaces = new Set(
    model.value.points
      .filter((point) => point.drawIndex === referenceIndex)
      .map((point) => point.space),
  );
  return [...referenceSpaces]
    .map((space) => {
      const precedent = model.value.points
        .filter(
          (point) => point.space === space && point.drawIndex < referenceIndex,
        )
        .sort((left, right) => right.drawIndex - left.drawIndex)[0];
      if (!precedent) return null;
      const topY = yForDraw(referenceIndex) + pointRadius;
      const bottomY = yForDraw(precedent.drawIndex) - pointRadius;
      return {
        space,
        x: xForSpace(space) - occurrenceStripWidth / 2,
        y: topY,
        height: Math.max(0, bottomY - topY),
      };
    })
    .filter(
      (strip): strip is {
        space: number;
        x: number;
        y: number;
        height: number;
      } => strip !== null && strip.height > 0,
    );
});

function xForSpace(space: number): number {
  return chartLeft + space * horizontalUnitSpacing;
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
    <div class="highlight-chart-scroll">
      <svg :width="svgWidth" height="42" class="highlight-chart-header" role="presentation">
        <g v-for="space in spaceUnits" :key="`header-space-${space}`">
          <circle
            :cx="xForSpace(space)"
            cy="21"
            :r="pointRadius"
            class="top-number-circle"
          />
          <text
            :class="{ compact: space >= 100 }"
            :x="xForSpace(space)"
            y="26"
            class="top-number-circle-label"
          >{{ space }}</text>
        </g>
      </svg>
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
          :width="chartPlotWidth + 22"
          height="28"
          class="current-reference-ribbon"
        />
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

        <rect
          v-for="band in drawSpaceBands"
          :key="`draw-space-band-${band.drawIndex}`"
          :x="xForSpace(band.first) - pointRadius"
          :y="yForDraw(band.drawIndex) - pointRadius"
          :width="xForSpace(band.last) - xForSpace(band.first) + pointRadius * 2"
          :height="pointRadius * 2"
          :rx="pointRadius"
          class="draw-occurrence-band"
        />

        <rect
          v-for="strip in occurrenceStrips"
          :key="`space-interval-${strip.space}`"
          :x="strip.x"
          :y="strip.y"
          :width="occurrenceStripWidth"
          :height="strip.height"
          class="occurrence-interval-strip"
          rx="4.5"
        />
        <rect
          v-for="strip in referencePrecedentStrips"
          :key="`space-precedent-${strip.space}`"
          :x="strip.x"
          :y="strip.y"
          :width="occurrenceStripWidth"
          :height="strip.height"
          class="reference-precedent-strip"
          rx="4.5"
        />

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
