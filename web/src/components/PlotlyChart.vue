<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import Plotly from "plotly.js-dist-min";
import type { FigureSpec } from "../types";

const props = defineProps<{
  figure: FigureSpec;
}>();

const chart = ref<HTMLElement | null>(null);
let resizeObserver: ResizeObserver | null = null;

async function render(): Promise<void> {
  await nextTick();
  if (!chart.value) return;
  await Plotly.react(
    chart.value,
    props.figure.data,
    { ...props.figure.layout, autosize: true },
    {
      responsive: true,
      displaylogo: false,
      scrollZoom: false,
      ...props.figure.config,
    },
  );
}

watch(() => props.figure, render, { deep: true });

onMounted(() => {
  void render();
  resizeObserver = new ResizeObserver(() => {
    if (chart.value) Plotly.Plots.resize(chart.value);
  });
  if (chart.value) resizeObserver.observe(chart.value);
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  if (chart.value) Plotly.purge(chart.value);
});
</script>

<template>
  <div ref="chart" class="plotly-chart" />
</template>
