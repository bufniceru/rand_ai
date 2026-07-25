<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import type { LastSeenDialogData } from "./types";
import LastSeenHighlightView from "./views/LastSeenHighlightView.vue";

const dialogData = ref<LastSeenDialogData | null>(null);
const errorMessage = ref("");
let unsubscribeData: (() => void) | null = null;

onMounted(async () => {
  document.title = "Last Seen Highlight — Rand AI";
  if (!window.randAiDesktop) {
    errorMessage.value =
      "Last Seen Highlight is available inside the Electron desktop application.";
    return;
  }
  unsubscribeData = window.randAiDesktop.onLastSeenData((data) => {
    dialogData.value = data;
  });
  try {
    dialogData.value = await window.randAiDesktop.getLastSeenData();
    if (!dialogData.value) {
      errorMessage.value = "Analyze a dataset before opening this dialog.";
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
});

onBeforeUnmount(() => {
  unsubscribeData?.();
});
</script>

<template>
  <main class="last-seen-dialog-shell">
    <LastSeenHighlightView
      v-if="dialogData"
      :history="dialogData.history"
    />
    <section v-else class="dialog-empty-state">
      <strong>Last Seen Highlight unavailable</strong>
      <p>{{ errorMessage || "Loading active draw history…" }}</p>
    </section>
  </main>
</template>
