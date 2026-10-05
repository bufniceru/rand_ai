<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { DrawEditorEntry, StrategyId } from "../types";

const props = defineProps<{
  anchor: HTMLElement;
  draw: DrawEditorEntry;
  number: number;
  result: { message: string; strategies: { id: StrategyId; name: string }[] };
}>();
const emit = defineEmits<{ enter: []; leave: []; dismiss: [] }>();
const panel = ref<HTMLElement>();
const left = ref(8), top = ref(8);
async function position() {
  await nextTick();
  if (!panel.value) return;
  const anchor = props.anchor.getBoundingClientRect();
  const bounds = panel.value.getBoundingClientRect();
  left.value = Math.max(8, Math.min(anchor.left, window.innerWidth - bounds.width - 8));
  const below = anchor.bottom + 8;
  let preferredTop = below;
  if (below + bounds.height > window.innerHeight - 8) {
    preferredTop = anchor.top - bounds.height - 8;
    if (preferredTop < 8) {
      preferredTop = anchor.top;
      if (anchor.right + bounds.width + 16 <= window.innerWidth) left.value = anchor.right + 8;
      else if (anchor.left - bounds.width - 8 >= 8) left.value = anchor.left - bounds.width - 8;
    }
  }
  top.value = Math.max(8, Math.min(preferredTop, window.innerHeight - bounds.height - 8));
}
function escape(event: KeyboardEvent) { if (event.key === "Escape") emit("dismiss"); }
watch(() => [props.anchor, props.number, props.result], position);
onMounted(() => {
  void position();
  window.addEventListener("resize", position);
  window.addEventListener("scroll", position, true);
  window.addEventListener("keydown", escape);
});
onBeforeUnmount(() => {
  window.removeEventListener("resize", position);
  window.removeEventListener("scroll", position, true);
  window.removeEventListener("keydown", escape);
});
</script>

<template>
  <Teleport to="body">
    <aside id="draw-number-hit-tooltip" ref="panel" role="tooltip" class="draw-number-hit-tooltip"
      :style="{ left: `${left}px`, top: `${top}px` }"
      @mouseenter="emit('enter')" @mouseleave="emit('leave')">
      <strong>Number {{ number }} · Draw {{ draw.index + 1 }}</strong>
      <small>{{ draw.date }}</small>
      <p v-if="result.message">{{ result.message }}</p>
      <template v-else>
        <p>{{ result.strategies.length }} successful {{ result.strategies.length === 1 ? 'strategy' : 'strategies' }} · Prior Top 6</p>
        <ul><li v-for="strategy in result.strategies" :key="strategy.id">{{ strategy.name }}</li></ul>
      </template>
    </aside>
  </Teleport>
</template>

<style scoped>
.draw-number-hit-tooltip {
  position: fixed;
  z-index: 10000;
  width: 360px;
  max-width: calc(100vw - 16px);
  max-height: min(420px, calc(100vh - 16px));
  overflow: auto;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--surface);
  color: var(--theme-text-primary, #fcfcfa);
  box-shadow: 0 8px 28px #0003;
}
strong, small { display: block; }
small { color: var(--text-muted); margin-top: 4px; }
p { margin: 10px 0; }
ul { margin: 0; padding-left: 20px; }
li + li { margin-top: 6px; }
</style>
