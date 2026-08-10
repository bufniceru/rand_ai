<script setup lang="ts">
import { computed, ref } from "vue";
import { buildLastSeenModel } from "../lib/lastSeen";
import type { HistoryDraw } from "../types";

const props = defineProps<{
  history: HistoryDraw[];
  drawCount: number;
  referenceDrawOffset: number;
}>();

const svgWidth = 1680;
const chartLeft = 70;
const chartTop = 30;
const chartBottom = 36;
const chartRight = 30;
const rowHeight = 30;
const pointRadius = 13.5;
const undrawnStripWidth = (pointRadius * 2) / 3;
const plotWidth = svgWidth - chartLeft - chartRight;

const selectedPoint = ref<{
  number: number;
  gapUntilReference: number;
  leftSpace: number;
  rightSpace: number;
} | null>(null);

const model = computed(() =>
  buildLastSeenModel(
    props.history,
    props.drawCount,
    props.referenceDrawOffset,
  ),
);
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
const yTicks = computed(() => {
  const step = Math.max(1, Math.floor(model.value.drawCount / 30));
  return rowDrawIndices.value
    .filter((_drawIndex, index) => index % step === 0)
    .map((drawIndex) => ({
      drawIndex,
      label: model.value.drawCount - drawIndex,
    }));
});
const drawNumberBands = computed(() => {
  const ranges = new Map<number, { first: number; last: number }>();
  for (const point of model.value.points) {
    const range = ranges.get(point.drawIndex);
    if (range) {
      range.first = Math.min(range.first, point.number);
      range.last = Math.max(range.last, point.number);
    } else {
      ranges.set(point.drawIndex, {
        first: point.number,
        last: point.number,
      });
    }
  }
  return [...ranges.entries()]
    .filter(([_drawIndex, range]) => range.first < range.last)
    .map(([drawIndex, range]) => ({ drawIndex, ...range }));
});
const undrawnStrips = computed(() => {
  const referenceIndex = model.value.referenceDrawIndex;
  if (referenceIndex === null) return [];
  return model.value.points
    .filter((point) => point.highlighted && point.drawIndex < referenceIndex)
    .map((point) => {
      const topY = yForDraw(referenceIndex) + pointRadius;
      const bottomY = yForDraw(point.drawIndex) - pointRadius;
      return {
        key: `${point.drawIndex}-${point.number}`,
        x: xForNumber(point.number) - undrawnStripWidth / 2,
        y: topY,
        height: Math.max(0, bottomY - topY),
      };
    })
    .filter((strip) => strip.height > 0);
});
const referencePrecedentStrips = computed(() => {
  const referenceIndex = model.value.referenceDrawIndex;
  if (referenceIndex === null) return [];
  const referenceNumbers = new Set(
    model.value.points
      .filter((point) => point.drawIndex === referenceIndex)
      .map((point) => point.number),
  );
  return [...referenceNumbers]
    .map((number) => {
      const precedent = model.value.points
        .filter((point) => point.number === number && point.drawIndex < referenceIndex)
        .sort((left, right) => right.drawIndex - left.drawIndex)[0];
      if (!precedent) return null;
      const topY = yForDraw(referenceIndex) + pointRadius;
      const bottomY = yForDraw(precedent.drawIndex) - pointRadius;
      return {
        key: `${referenceIndex}-${precedent.drawIndex}-${number}`,
        x: xForNumber(number) - undrawnStripWidth / 2,
        y: topY,
        height: Math.max(0, bottomY - topY),
      };
    })
    .filter(
      (strip): strip is { key: string; x: number; y: number; height: number } =>
        strip !== null && strip.height > 0,
    );
});

function xForNumber(number: number): number {
  return chartLeft + ((number - 1) / 48) * plotWidth;
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

function showPoint(
  number: number,
  drawIndex: number,
  leftSpace: number,
  rightSpace: number,
): void {
  const referenceIndex = model.value.referenceDrawIndex ?? drawIndex;
  selectedPoint.value = {
    number,
    gapUntilReference: Math.max(0, referenceIndex - drawIndex - 1),
    leftSpace,
    rightSpace,
  };
}
</script>

<template>
  <section class="workspace-view last-seen-view last-seen-number-view">
    <div class="highlight-chart-scroll">
      <svg :width="svgWidth" height="42" class="highlight-chart-header" role="presentation">
        <g v-for="number in 49" :key="`header-${number}`">
          <circle
            :cx="xForNumber(number)"
            cy="21"
            :r="pointRadius"
            class="top-number-circle"
          />
          <text
            :x="xForNumber(number)"
            y="26"
            class="top-number-circle-label"
          >{{ number }}</text>
        </g>
      </svg>
      <svg :height="chartHeight" :width="svgWidth" class="highlight-chart" role="img">
        <text class="axis-label" :x="svgWidth / 2" :y="chartHeight - 10">Number</text>
        <text
          class="axis-label"
          :x="18"
          :y="chartHeight / 2"
          transform="rotate(-90, 18, 240)"
        >Draw Index</text>

        <line
          v-for="number in 49"
          :key="`v-${number}`"
          :class="{ major: number % 5 === 0 }"
          class="vertical-guide"
          :x1="xForNumber(number)"
          :x2="xForNumber(number)"
          :y1="chartTop - 10"
          :y2="chartTop + plotHeight + 10"
        />
        <rect
          v-if="model.referenceDrawIndex !== null"
          :x="xForNumber(1) - 11"
          :y="yForDraw(model.referenceDrawIndex) - 14"
          :width="plotWidth + 22"
          height="28"
          class="current-reference-ribbon"
        />
        <g v-for="drawIndex in rowDrawIndices" :key="`h-${drawIndex}`">
          <line
            :class="{ major: (model.drawCount - drawIndex) % 5 === 0 }"
            class="horizontal-guide"
            :x1="xForNumber(1)"
            :x2="xForNumber(49)"
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
          v-for="number in 49"
          :key="`bottom-${number}`"
          :class="{ major: number % 5 === 0 }"
          :x="xForNumber(number)"
          :y="chartTop + plotHeight + 28"
          class="tick-label x-tick"
        >{{ number }}</text>

        <rect
          v-for="band in drawNumberBands"
          :key="`draw-number-band-${band.drawIndex}`"
          :x="xForNumber(band.first) - pointRadius"
          :y="yForDraw(band.drawIndex) - pointRadius"
          :width="xForNumber(band.last) - xForNumber(band.first) + pointRadius * 2"
          :height="pointRadius * 2"
          :rx="pointRadius"
          class="draw-occurrence-band"
        />

        <rect
          v-for="strip in undrawnStrips"
          :key="`undrawn-${strip.key}`"
          :height="strip.height"
          :width="undrawnStripWidth"
          :x="strip.x"
          :y="strip.y"
          class="occurrence-interval-strip"
          rx="4.5"
        />
        <rect
          v-for="strip in referencePrecedentStrips"
          :key="`precedent-${strip.key}`"
          :height="strip.height"
          :width="undrawnStripWidth"
          :x="strip.x"
          :y="strip.y"
          class="reference-precedent-strip"
          rx="4.5"
        />

        <g
          v-for="point in model.points"
          :key="`${point.drawIndex}-${point.number}`"
          class="highlight-point"
          @click="showPoint(point.number, point.drawIndex, point.leftSpace, point.rightSpace)"
        >
          <circle
            :class="pointClass(point.drawIndex, point.highlighted)"
            :cx="xForNumber(point.number)"
            :cy="yForDraw(point.drawIndex)"
            :r="pointRadius"
          />
          <text
            :x="xForNumber(point.number)"
            :y="yForDraw(point.drawIndex) + 5"
            class="point-label"
          >{{ point.gap }}</text>
          <text
            :x="xForNumber(point.number)"
            :y="yForDraw(point.drawIndex) + 13"
            class="point-space-label"
          >L {{ point.leftSpace }} · R {{ point.rightSpace }}</text>
        </g>
      </svg>
    </div>

    <div
      v-if="selectedPoint"
      class="number-popup-backdrop"
      role="presentation"
      @click.self="selectedPoint = null"
    >
      <section class="number-popup">
        <p>Selected number</p>
        <strong>{{ selectedPoint.number }}</strong>
        <span>Gap until reference: {{ selectedPoint.gapUntilReference }}</span>
        <span>Left space: {{ selectedPoint.leftSpace }}</span>
        <span>Right space: {{ selectedPoint.rightSpace }}</span>
        <button type="button" @click="selectedPoint = null">Close</button>
      </section>
    </div>
  </section>
</template>
