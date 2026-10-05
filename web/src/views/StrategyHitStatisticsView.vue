<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { AnalysisPayload, StrategyId } from "../types";
import { contributingDraws, strategyHitStatistics, type HitHistoryScope, type RelationshipMeasure } from "../lib/strategyHitStatistics";

const props = defineProps<{ analysis: AnalysisPayload }>();
const scope = ref<HitHistoryScope>("all");
const measure = ref<RelationshipMeasure>("sharedNumbers");
const selectedDraw = ref<number>();
const selectedPair = ref("");
const sort = ref<RelationshipMeasure | "sharedDraws" | "name">("sharedNumbers");
const descending = ref(true);
const stats = computed(() => strategyHitStatistics(props.analysis.predictionAuditHistory,
  props.analysis.strategyEfficacyHistory, props.analysis.options.enabledStrategies, scope.value));
watch(() => props.analysis, () => {
  selectedDraw.value = stats.value.draws.at(-1)?.record.targetDrawNumber;
  selectedPair.value = "";
});
watch(measure, value => { sort.value = value; descending.value = true; });
watch(() => stats.value.draws, draws => {
  if (!draws.some(draw => draw.record.targetDrawNumber === selectedDraw.value)) selectedDraw.value = draws.at(-1)?.record.targetDrawNumber;
}, { immediate: true });
const draw = computed(() => stats.value.draws.find(draw => draw.record.targetDrawNumber === selectedDraw.value));
const totals = computed(() => [...stats.value.strategies].sort((a, b) => b.hits - a.hits || a.name.localeCompare(b.name)));
function pairKey(left: StrategyId, right: StrategyId) { return [left, right].sort().join(":"); }
const pairMap = computed(() => new Map(stats.value.pairs.map(pair => [pairKey(pair.left, pair.right), pair])));
const pairs = computed(() => [...stats.value.pairs].sort((a, b) => {
  const delta = sort.value === "name" ? `${stats.value.names.get(a.left)} / ${stats.value.names.get(a.right)}`.localeCompare(`${stats.value.names.get(b.left)} / ${stats.value.names.get(b.right)}`) : a[sort.value] - b[sort.value];
  return descending.value ? -delta : delta;
}));
const pair = computed(() => pairMap.value.get(selectedPair.value));
const details = computed(() => pair.value ? contributingDraws(stats.value.draws, pair.value, measure.value) : []);
const maxValue = computed(() => Math.max(1, ...stats.value.pairs.map(pair => pair[measure.value])));
function cell(left: StrategyId, right: StrategyId) { return pairMap.value.get(pairKey(left, right))?.[measure.value] ?? 0; }
function sortBy(key: typeof sort.value) {
  descending.value = sort.value === key ? !descending.value : key !== "name";
  sort.value = key;
}
</script>

<template>
  <section class="workspace-view strategy-hit-statistics-view">
    <header class="prediction-analysis-header">
      <div><h1>Strategy Hit Statistics</h1><p>A hit is a winning number included in a strategy’s prior Top 6. Relationships describe observed hits.</p></div>
      <label>History <select v-model="scope"><option value="all">All evaluated draws</option><option :value="100">Latest 100</option><option :value="250">Latest 250</option><option :value="500">Latest 500</option></select></label>
    </header>
    <p v-if="stats.draws.length">{{ stats.draws.length }} evaluated draws · Draw {{ stats.draws[0].record.targetDrawNumber }}–{{ stats.draws.at(-1)?.record.targetDrawNumber }}</p>
    <p v-else>No evaluated draws available. Enable strategies and load a dataset with prior draws.</p>
    <section v-if="draw" class="prediction-history-panel">
      <h2>Draw details</h2>
      <label>Draw <select v-model="selectedDraw"><option v-for="item in [...stats.draws].reverse()" :key="item.record.targetDrawNumber" :value="item.record.targetDrawNumber">{{ item.record.targetDrawNumber }}{{ item.record.date ? ` · ${item.record.date}` : '' }}</option></select></label>
      <p>Predicted from draw {{ draw.record.referenceDrawNumber }}.</p>
      <div class="hit-statistics-columns">
        <table><thead><tr><th>Winning number</th><th>Successful strategies</th></tr></thead><tbody><tr v-for="item in draw.record.numbers" :key="item.number"><th>{{ item.number }}</th><td>{{ item.strategies.map(strategy => stats.names.get(strategy.id)).join(', ') || 'None' }}</td></tr></tbody></table>
        <table><thead><tr><th>Strategy</th><th>Hits</th><th>Matched numbers</th></tr></thead><tbody><tr v-for="strategy in stats.strategies" :key="strategy.id"><th>{{ strategy.name }}</th><td>{{ draw.matches[strategy.id]?.length ?? 0 }}</td><td>{{ draw.matches[strategy.id]?.join(', ') || 'None' }}</td></tr></tbody></table>
      </div>
    </section>
    <section class="prediction-history-panel"><h2>Strategy totals</h2><div class="hit-statistics-scroll"><table><thead><tr><th>Strategy</th><th>Total hits</th><th>Draws with hits</th><th>Evaluated draws</th><th>Average hits</th><th v-for="count in 7" :key="count">{{ count - 1 }} hits</th></tr></thead><tbody><tr v-for="strategy in totals" :key="strategy.id"><th>{{ strategy.name }}</th><td>{{ strategy.hits }}</td><td>{{ strategy.hitDraws }}</td><td>{{ strategy.evaluatedDraws }}</td><td>{{ strategy.average.toFixed(2) }}</td><td v-for="(count, index) in strategy.buckets" :key="index">{{ count }}</td></tr></tbody></table></div></section>
    <section class="prediction-history-panel">
      <h2>Strategy relationships</h2>
      <label>Measure <select v-model="measure"><option value="sharedNumbers">Shared winning numbers</option><option value="bothHitDraws">Both hit in the same draw</option></select></label>
      <p>{{ measure === 'sharedNumbers' ? 'Each shared draw-and-number event counts once per pair.' : 'Each draw where both strategies hit counts once, including hits on different numbers.' }} Click a cell or pair to inspect contributing draws.</p>
      <div class="hit-statistics-scroll"><table class="hit-statistics-heatmap"><thead><tr><th>Strategy</th><th v-for="strategy in stats.strategies" :key="strategy.id">{{ strategy.name }}</th></tr></thead><tbody><tr v-for="left in stats.strategies" :key="left.id"><th>{{ left.name }}</th><td v-for="right in stats.strategies" :key="right.id"><span v-if="left.id === right.id">—</span><button v-else :aria-label="`${left.name} and ${right.name}: ${cell(left.id, right.id)}`" :style="{ background: `color-mix(in srgb, var(--blue) ${Math.round(10 + 65 * cell(left.id, right.id) / maxValue)}%, var(--surface))` }" @click="selectedPair = pairKey(left.id, right.id)">{{ cell(left.id, right.id) }}</button></td></tr></tbody></table></div>
      <div class="hit-statistics-scroll hit-statistics-pairs"><table><thead><tr><th><button @click="sortBy('name')">Strategy pair ↕</button></th><th><button @click="sortBy('sharedNumbers')">Shared numbers ↕</button></th><th><button @click="sortBy('sharedDraws')">Draws with shared hits ↕</button></th><th><button @click="sortBy('bothHitDraws')">Both hit draws ↕</button></th></tr></thead><tbody><tr v-for="item in pairs" :key="pairKey(item.left, item.right)"><th><button @click="selectedPair = pairKey(item.left, item.right)">{{ stats.names.get(item.left) }} / {{ stats.names.get(item.right) }}</button></th><td>{{ item.sharedNumbers }}</td><td>{{ item.sharedDraws }}</td><td>{{ item.bothHitDraws }}</td></tr></tbody></table></div>
      <div v-if="pair"><h3>{{ stats.names.get(pair.left) }} / {{ stats.names.get(pair.right) }}</h3><p v-if="!details.length">No contributing draws for this measure.</p><div v-else class="hit-statistics-scroll hit-statistics-pairs"><table><thead><tr><th>Draw</th><th>{{ stats.names.get(pair.left) }}</th><th>{{ stats.names.get(pair.right) }}</th><th>Shared numbers</th></tr></thead><tbody><tr v-for="item in details" :key="item.record.targetDrawNumber"><th><button @click="selectedDraw = item.record.targetDrawNumber">{{ item.record.targetDrawNumber }}</button></th><td>{{ item.matches[pair.left]?.join(', ') }}</td><td>{{ item.matches[pair.right]?.join(', ') }}</td><td>{{ item.matches[pair.left]?.filter(number => item.matches[pair!.right]?.includes(number)).join(', ') || 'None' }}</td></tr></tbody></table></div></div>
    </section>
  </section>
</template>

<style scoped>
.strategy-hit-statistics-view { display: grid; gap: 20px; }
.hit-statistics-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; overflow: auto; }
.hit-statistics-scroll { overflow: auto; margin-top: 16px; }
.hit-statistics-pairs { max-height: 420px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); }
thead th { white-space: nowrap; }
button, select { font: inherit; cursor: pointer; color: inherit; }
button { border: 1px solid var(--line); border-radius: 4px; background: transparent; padding: 5px 8px; }
.hit-statistics-heatmap td { padding: 2px; text-align: center; }
.hit-statistics-heatmap button { width: 100%; min-width: 40px; }
@media (max-width: 1000px) { .hit-statistics-columns { grid-template-columns: 1fr; } }
</style>
