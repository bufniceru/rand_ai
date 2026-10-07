<script setup lang="ts">
import type { StrategyId } from "../types";
defineProps<{ rows: { id: StrategyId; name: string; color: string; average: number; lift: number }[];
  selected: Set<StrategyId> }>();
const emit = defineEmits<{ toggle: [id: StrategyId, visible: boolean] }>();
</script>

<template>
  <div class="strategy-timeline-legend" role="group" aria-label="Visible strategies">
    <label v-for="row in rows" :key="row.id" :class="{ selected: selected.has(row.id) }">
      <input type="checkbox" :checked="selected.has(row.id)"
        @change="emit('toggle', row.id, ($event.target as HTMLInputElement).checked)">
      <i :style="{ '--strategy-line-color': row.color }" />
      <span><strong>{{ row.name }}</strong>
        <small>{{ row.average.toFixed(3) }} · {{ row.lift >= 0 ? '+' : '' }}{{ row.lift.toFixed(3) }} lift</small>
      </span>
    </label>
  </div>
</template>
