<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import {
  filterApplicationCommands,
  type AppCommand,
} from "../lib/commands";

const props = defineProps<{
  commands: readonly AppCommand[];
  hasDataset: boolean;
}>();

const emit = defineEmits<{
  cancel: [];
  execute: [commandId: string];
}>();

const search = ref("");
const searchInput = ref<HTMLInputElement | null>(null);
const activeIndex = ref(0);
const filteredCommands = computed(() =>
  filterApplicationCommands(props.commands, search.value),
);

function disabledReason(command: AppCommand): string | null {
  return command.disabledReason({ hasDataset: props.hasDataset });
}

function moveSelection(direction: -1 | 1): void {
  const count = filteredCommands.value.length;
  if (count === 0) return;
  activeIndex.value = (activeIndex.value + direction + count) % count;
}

function execute(command: AppCommand | undefined): void {
  if (!command || disabledReason(command)) return;
  emit("execute", command.id);
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") {
    event.preventDefault();
    emit("cancel");
    return;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    moveSelection(event.key === "ArrowDown" ? 1 : -1);
    return;
  }
  if (event.key === "Enter") {
    event.preventDefault();
    execute(filteredCommands.value[activeIndex.value]);
  }
}

watch(filteredCommands, () => {
  activeIndex.value = 0;
});

onMounted(() => searchInput.value?.focus());
</script>

<template>
  <div class="command-palette-backdrop" role="presentation" @click.self="emit('cancel')">
    <section
      class="command-palette"
      role="dialog"
      aria-modal="true"
      aria-labelledby="command-palette-title"
      @keydown="handleKeydown"
    >
      <h2 id="command-palette-title">Command Palette</h2>
      <label class="command-palette-search">
        <span aria-hidden="true">&gt;</span>
        <input
          ref="searchInput"
          v-model="search"
          type="search"
          placeholder="Type a command"
          aria-label="Search commands"
          autocomplete="off"
        >
      </label>
      <ul role="listbox" aria-label="Available commands">
        <li v-for="(command, index) in filteredCommands" :key="command.id">
          <button
            type="button"
            role="option"
            :aria-selected="index === activeIndex"
            :aria-disabled="Boolean(disabledReason(command))"
            :class="{
              active: index === activeIndex,
              disabled: Boolean(disabledReason(command)),
            }"
            @mouseenter="activeIndex = index"
            @click="execute(command)"
          >
            <span><b>{{ command.category }}:</b> {{ command.title }}</span>
            <small v-if="disabledReason(command)">{{ disabledReason(command) }}</small>
          </button>
        </li>
        <li v-if="filteredCommands.length === 0" class="command-palette-empty">
          No matching commands
        </li>
      </ul>
      <footer><span>↑↓ Navigate</span><span>Enter Run</span><span>Esc Close</span></footer>
    </section>
  </div>
</template>
