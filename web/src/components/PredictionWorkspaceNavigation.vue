<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";

type WorkspaceDestination = "predictions" | "possible-draw" | "draw-history";

const props = defineProps<{
  active: WorkspaceDestination;
}>();

const errorMessage = ref("");
const destinations: {
  id: WorkspaceDestination;
  label: string;
  shortcut: string;
}[] = [
  { id: "predictions", label: "Predictions", shortcut: "Alt+1" },
  { id: "possible-draw", label: "Possible Draw", shortcut: "Alt+2" },
  { id: "draw-history", label: "Draw History", shortcut: "Alt+3" },
];

async function navigate(destination: WorkspaceDestination): Promise<void> {
  if (destination === props.active) return;
  if (!window.randAiDesktop) {
    errorMessage.value = "Navigation is available inside the Electron application.";
    return;
  }
  errorMessage.value = "";
  try {
    if (destination === "predictions") {
      await window.randAiDesktop.openCombinedPredictionDialog();
    } else if (destination === "possible-draw") {
      await window.randAiDesktop.openPossibleDrawDialog();
    } else {
      await window.randAiDesktop.openDrawEditorDialog();
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

function handleShortcut(event: KeyboardEvent): void {
  if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  const destination = {
    Digit1: "predictions",
    Digit2: "possible-draw",
    Digit3: "draw-history",
  }[event.code] as WorkspaceDestination | undefined;
  if (!destination) return;
  event.preventDefault();
  void navigate(destination);
}

onMounted(() => window.addEventListener("keydown", handleShortcut));
onBeforeUnmount(() => window.removeEventListener("keydown", handleShortcut));
</script>

<template>
  <nav
    class="prediction-workspace-navigation"
    :class="{ 'has-context-controls': $slots.controls }"
    aria-label="Prediction workspace"
  >
    <div class="prediction-workspace-navigation-label">
      <span>Prediction workspace</span>
      <small>Switch views without closing your work</small>
    </div>
    <div class="prediction-workspace-navigation-buttons">
      <button
        v-for="destination in destinations"
        :key="destination.id"
        type="button"
        :class="{ active: destination.id === active }"
        :aria-current="destination.id === active ? 'page' : undefined"
        :title="`${destination.label} (${destination.shortcut})`"
        @click="navigate(destination.id)"
      >
        <span>{{ destination.label }}</span>
        <kbd>{{ destination.shortcut }}</kbd>
      </button>
    </div>
    <div
      v-if="$slots.controls"
      class="prediction-workspace-navigation-controls"
    >
      <slot name="controls" />
    </div>
    <p v-if="errorMessage" role="alert">{{ errorMessage }}</p>
  </nav>
</template>
