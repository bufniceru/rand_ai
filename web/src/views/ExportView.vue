<script setup lang="ts">
import type { AnalysisPayload } from "../types";

defineProps<{
  analysis: AnalysisPayload;
  exporting: boolean;
  exportMessage: string;
}>();

defineEmits<{
  export: [];
}>();
</script>

<template>
  <section class="workspace-view export-view">
    <header class="view-header">
      <div><p class="eyebrow">Portable results</p><h2>Export analysis</h2></div>
      <p>Save auditable metadata and every compact statistics table as CSV files.</p>
    </header>
    <article class="export-card">
      <div class="export-icon">ZIP</div>
      <div>
        <h3>Draws statistics archive</h3>
        <p>
          Includes metadata and tables only for the report plugins currently
          checked in the Reports menu.
        </p>
        <p class="muted">
          {{ analysis.options.enabledReports.length }} enabled reports ·
          {{ Object.keys(analysis.tables).length }} tables ·
          {{ analysis.dataset.drawCount.toLocaleString() }} draws
        </p>
      </div>
      <button class="button primary" :disabled="exporting" type="button" @click="$emit('export')">
        {{ exporting ? "Exporting…" : "Save ZIP archive" }}
      </button>
    </article>
    <p v-if="exportMessage" class="export-message">{{ exportMessage }}</p>
  </section>
</template>
