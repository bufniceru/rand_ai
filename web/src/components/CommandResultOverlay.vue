<script setup lang="ts">
import { onMounted, ref } from "vue";
import type { CommandResultOverlayState } from "../lib/commands";
import PlotlyChart from "./PlotlyChart.vue";
import GapsView from "../views/GapsView.vue";

defineProps<{ state: CommandResultOverlayState }>();
defineEmits<{ close: [] }>();

const dialog = ref<HTMLElement | null>(null);
onMounted(() => dialog.value?.focus());
</script>

<template>
  <section
    ref="dialog"
    class="command-result-overlay"
    role="dialog"
    aria-modal="true"
    tabindex="-1"
    @keydown.esc.stop.prevent="$emit('close')"
  >
    <header>
      <div>
        <p class="eyebrow">Command result</p>
        <h1>{{ state.status === "ready" ? state.result.title : state.title }}</h1>
        <p v-if="state.status === 'ready'">{{ state.result.subtitle }}</p>
      </div>
      <span>Press Esc to return</span>
    </header>
    <div v-if="state.status === 'loading'" class="command-result-message" role="status">
      <span class="spinner" />
      <strong>Calculating against the complete database…</strong>
    </div>
    <div v-else-if="state.status === 'error'" class="command-result-message error" role="alert">
      <strong>Command failed</strong>
      <p>{{ state.message }}</p>
    </div>
    <div v-else-if="state.result.kind === 'gap-statistics'" class="command-result-report">
      <GapsView :table="state.result.table" :figures="state.result.figures" />
    </div>
    <article v-else class="command-result-figure">
      <PlotlyChart :figure="state.result.figure" />
    </article>
  </section>
</template>

<style scoped>
.command-result-report {
  min-height: 0;
  overflow: auto;
}
</style>
