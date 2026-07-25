<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import PredictionWorkspaceNavigation from "./components/PredictionWorkspaceNavigation.vue";
import type { DrawEditorData, DrawEditorEntry } from "./types";

type EditorMode = "view" | "add" | "edit";
type DrawVisualization = "grid" | "circle";

const data = ref<DrawEditorData | null>(null);
const currentIndex = ref(0);
const mode = ref<EditorMode>("view");
const visualization = ref<DrawVisualization>("grid");
const editDate = ref("");
const editNumbers = ref<number[]>([]);
const originalDate = ref("");
const loading = ref(true);
const saving = ref(false);
const message = ref("");
const errorMessage = ref("");

const draws = computed(() => data.value?.draws ?? []);
const currentDraw = computed<DrawEditorEntry | null>(
  () => draws.value[currentIndex.value] ?? null,
);
const displayedNumbers = computed(() =>
  mode.value === "view" ? currentDraw.value?.numbers ?? [] : editNumbers.value,
);
const displayedSet = computed(() => new Set(displayedNumbers.value));
const canSave = computed(
  () => /^\d{4}-\d{2}-\d{2}$/.test(editDate.value) && editNumbers.value.length === 6,
);
const circleNumbers = Array.from({ length: 49 }, (_value, index) => {
  const number = index + 1;
  const angle = (index / 49) * Math.PI * 2 - Math.PI / 2;
  return {
    number,
    left: `${50 + Math.cos(angle) * 44}%`,
    top: `${50 + Math.sin(angle) * 44}%`,
  };
});

function navigate(index: number): void {
  if (!draws.value.length) return;
  currentIndex.value = Math.min(Math.max(Math.trunc(index), 0), draws.value.length - 1);
  mode.value = "view";
  message.value = "";
  errorMessage.value = "";
}

function beginAdd(): void {
  mode.value = "add";
  originalDate.value = "";
  editDate.value = new Date().toISOString().slice(0, 10);
  editNumbers.value = [];
  message.value = "";
  errorMessage.value = "";
}

function beginEdit(): void {
  if (!currentDraw.value) return;
  mode.value = "edit";
  originalDate.value = currentDraw.value.date;
  editDate.value = currentDraw.value.date;
  editNumbers.value = [...currentDraw.value.numbers];
  message.value = "";
  errorMessage.value = "";
}

function cancelEdit(): void {
  mode.value = "view";
  errorMessage.value = "";
}

function toggleNumber(number: number): void {
  if (mode.value === "view") return;
  if (displayedSet.value.has(number)) {
    editNumbers.value = editNumbers.value.filter((item) => item !== number);
  } else if (editNumbers.value.length < 6) {
    editNumbers.value = [...editNumbers.value, number].sort((left, right) => left - right);
  }
}

async function save(): Promise<void> {
  if (!window.randAiDesktop || !canSave.value) return;
  saving.value = true;
  message.value = "";
  errorMessage.value = "";
  try {
    const savedDate = editDate.value;
    data.value = await window.randAiDesktop.saveDraw({
      date: String(savedDate),
      numbers: [...editNumbers.value],
      ...(mode.value === "edit" ? { originalDate: originalDate.value } : {}),
    });
    currentIndex.value = Math.max(
      data.value.draws.findIndex((draw) => draw.date === savedDate),
      0,
    );
    mode.value = "view";
    message.value = "YAML saved first; the equivalent pickle was rebuilt. Reanalysis is ready in the main window.";
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    saving.value = false;
  }
}

onMounted(async () => {
  document.title = "Draw History — Rand AI";
  if (!window.randAiDesktop) {
    errorMessage.value = "Draw History is available inside the Electron application.";
    loading.value = false;
    return;
  }
  try {
    data.value = await window.randAiDesktop.getDrawEditorData();
    currentIndex.value = Math.max(data.value.draws.length - 1, 0);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <main class="draw-editor-shell">
    <PredictionWorkspaceNavigation active="draw-history" />
    <section class="draw-editor-window">
      <header class="draw-editor-header">
        <div>
          <span>YAML-managed history</span>
          <h1>Draw History</h1>
          <p v-if="data">{{ data.yamlPath }}</p>
        </div>
        <div class="draw-editor-actions">
          <button type="button" :disabled="loading || mode !== 'view'" @click="beginAdd">New draw</button>
          <button type="button" :disabled="!currentDraw || mode !== 'view'" @click="beginEdit">Edit draw</button>
          <button v-if="mode !== 'view'" type="button" @click="cancelEdit">Cancel</button>
          <button v-if="mode !== 'view'" class="primary" type="button" :disabled="!canSave || saving" @click="save">
            {{ saving ? "Saving…" : "Save draw" }}
          </button>
        </div>
      </header>

      <nav class="draw-editor-navigation" aria-label="Draw navigation">
        <button type="button" :disabled="currentIndex === 0 || mode !== 'view'" @click="navigate(0)">|&lt;</button>
        <button type="button" :disabled="currentIndex === 0 || mode !== 'view'" @click="navigate(currentIndex - 1)">&lt;</button>
        <label>
          <span>Draw</span>
          <input
            :value="currentIndex + 1"
            type="number"
            min="1"
            :max="draws.length"
            :disabled="mode !== 'view'"
            @change="navigate(Number(($event.target as HTMLInputElement).value) - 1)"
          >
          <small>of {{ draws.length }}</small>
        </label>
        <button type="button" :disabled="currentIndex >= draws.length - 1 || mode !== 'view'" @click="navigate(currentIndex + 1)">&gt;</button>
        <button type="button" :disabled="currentIndex >= draws.length - 1 || mode !== 'view'" @click="navigate(draws.length - 1)">&gt;|</button>
        <label class="draw-date-field">
          <span>Date</span>
          <input v-if="mode !== 'view'" v-model="editDate" type="date">
          <strong v-else>{{ currentDraw?.date ?? "—" }}</strong>
        </label>
        <div class="draw-selection-count">
          <span>Selected</span>
          <strong>{{ displayedNumbers.length }} / 6</strong>
        </div>
      </nav>

      <nav class="draw-editor-view-tabs" role="tablist" aria-label="Draw visualization">
        <button
          type="button"
          role="tab"
          :aria-selected="visualization === 'grid'"
          :class="{ active: visualization === 'grid' }"
          @click="visualization = 'grid'"
        >
          7×7 Grid
        </button>
        <button
          type="button"
          role="tab"
          :aria-selected="visualization === 'circle'"
          :class="{ active: visualization === 'circle' }"
          @click="visualization = 'circle'"
        >
          PyLotto Circle
        </button>
      </nav>

      <section v-if="!loading && data" class="draw-editor-content">
        <div
          v-if="visualization === 'grid'"
          class="draw-editor-grid"
          role="grid"
          aria-label="Draw numbers in a 7 by 7 grid"
        >
          <button
            v-for="number in 49"
            :key="number"
            type="button"
            role="gridcell"
            :class="{ selected: displayedSet.has(number), editable: mode !== 'view' }"
            :aria-pressed="displayedSet.has(number)"
            @click="toggleNumber(number)"
          >
            {{ number }}
          </button>
        </div>
        <div
          v-else
          class="draw-editor-circle"
          role="group"
          aria-label="Draw numbers arranged around a PyLotto circle"
        >
          <div class="draw-editor-circle-rings" aria-hidden="true" />
          <div class="draw-editor-circle-center" aria-hidden="true">
            <span>Draw</span>
            <strong>{{ currentIndex + 1 }}</strong>
            <small>{{ displayedNumbers.length }} / 6 selected</small>
          </div>
          <button
            v-for="entry in circleNumbers"
            :key="entry.number"
            type="button"
            :class="{
              selected: displayedSet.has(entry.number),
              editable: mode !== 'view',
            }"
            :style="{ left: entry.left, top: entry.top }"
            :aria-pressed="displayedSet.has(entry.number)"
            @click="toggleNumber(entry.number)"
          >
            {{ entry.number }}
          </button>
        </div>
        <div class="draw-editor-summary">
          <span>{{ mode === "view" ? "Draw numbers" : mode === "add" ? "New draw" : "Editing draw" }}</span>
          <div>
            <strong v-for="number in displayedNumbers" :key="number">{{ number }}</strong>
            <i v-for="slot in 6 - displayedNumbers.length" :key="`empty-${slot}`">—</i>
          </div>
          <p v-if="mode !== 'view'">Select exactly six numbers, set the date, then save.</p>
          <p v-else>Use the navigation controls to browse the complete YAML draw history.</p>
        </div>
      </section>

      <section v-else-if="loading" class="dialog-empty-state">
        <strong>Loading draw history…</strong>
      </section>
      <p v-if="message" class="draw-editor-status success">{{ message }}</p>
      <p v-if="errorMessage" class="draw-editor-status error">{{ errorMessage }}</p>
    </section>
  </main>
</template>
