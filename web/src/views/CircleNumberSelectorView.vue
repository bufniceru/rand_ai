<script setup lang="ts">
import { computed } from "vue";
import type { HistoryDraw } from "../types";
import { circleNumbers, CIRCLE_CANVAS_SIZE, NUMBER_BALL_RADIUS } from "../lib/circleNumbers";
import { POSSIBLE_DRAW_SELECTION_HELP, usePossibleDrawNumberControls } from "../lib/possibleDrawNumberControls";

const props = defineProps<{ history: HistoryDraw[]; targetDrawNumber: number | null }>();
const selectionEditable = computed(() => props.history.length > 0 && props.targetDrawNumber !== null);
const { numberActionMessage, numberState, handleHeaderClick, handleHeaderKeydown } =
  usePossibleDrawNumberControls(selectionEditable);
</script>

<template>
  <section class="workspace-view circle-number-view">
    <h2>Circle <span v-if="targetDrawNumber !== null">· Target draw {{ targetDrawNumber }}</span></h2>
    <p class="last-seen-selection-help">{{ selectionEditable
      ? POSSIBLE_DRAW_SELECTION_HELP
      : "Load a dataset with history to edit Possible Draw selections." }}</p>
    <p v-if="numberActionMessage" role="status" class="last-seen-selection-message">{{ numberActionMessage }}</p>
    <div class="circle-number-scroll">
      <svg :width="CIRCLE_CANVAS_SIZE" :height="CIRCLE_CANVAS_SIZE"
        class="circle-number-canvas" role="group" aria-label="Circle Possible Draw number selection">
        <g v-for="point in circleNumbers" :key="point.number"
          class="last-seen-header-number"
          :class="[`possible-${numberState(point.number)}`, { 'possible-readonly': !selectionEditable }]"
          role="button" tabindex="0" :aria-disabled="!selectionEditable"
          :aria-label="`Number ${point.number}; Possible Draw state ${numberState(point.number)}${selectionEditable ? '' : '; read-only without dataset history'}`"
          @click="handleHeaderClick($event, point.number)"
          @keydown="handleHeaderKeydown($event, point.number)">
          <circle :cx="point.x" :cy="point.y" :r="NUMBER_BALL_RADIUS" class="top-number-circle" />
          <text :x="point.x" :y="point.y + 5" class="top-number-circle-label">{{ point.number }}</text>
          <text v-if="numberState(point.number) !== 'neutral'" :x="point.x + 10" :y="point.y - 9"
            class="last-seen-selection-badge" aria-hidden="true">{{ numberState(point.number) === 'candidate' ? 'C' : numberState(point.number) === 'fixed' ? '🔒' : '×' }}</text>
        </g>
      </svg>
    </div>
  </section>
</template>
