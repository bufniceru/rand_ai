<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type {
  CombinedPredictionDialogData,
  RelationshipEdge,
  StrategyId,
  StrategyNumberPrediction,
  StrategyPrediction,
} from "./types";

interface DrawPlan {
  id: string;
  name: string;
  selected: number[];
  dropped: number[];
  uncertain: number[];
}

interface RelatedSuggestion {
  number: number;
  lift: number;
  hits: number;
  score: number;
}

const storageKey = "rand-ai.possible-draw.plans.v2";
const dialogData = ref<CombinedPredictionDialogData | null>(null);
const plans = ref<DrawPlan[]>([]);
const activePlanId = ref("");
const selectedNumbers = ref<number[]>([]);
const droppedNumbers = ref<number[]>([]);
const uncertainNumbers = ref<number[]>([]);
const focusedNumber = ref<number | null>(null);
const showLastDraw = ref(false);
const showLastSeen = ref(true);
const lastSeenIndex = ref(0);
const errorMessage = ref("");
let clickTimer: ReturnType<typeof setTimeout> | null = null;
let unsubscribeData: (() => void) | null = null;

const latestSuite = computed(() => dialogData.value?.predictionSuites.at(-1) ?? null);
const strategies = computed(() => latestSuite.value?.strategies ?? []);
const strategyById = computed(
  () => new Map(strategies.value.map((strategy) => [strategy.id, strategy])),
);
const selectedSet = computed(() => new Set(selectedNumbers.value));
const droppedSet = computed(() => new Set(droppedNumbers.value));
const uncertainSet = computed(() => new Set(uncertainNumbers.value));
const lastDrawSet = computed(
  () => new Set(dialogData.value?.possibleDraw.lastDrawNumbers ?? []),
);
const lastSeenRows = computed(() => dialogData.value?.possibleDraw.lastSeenRows ?? []);
const highlightedLastSeen = computed(() => lastSeenRows.value[lastSeenIndex.value] ?? null);
const activePlan = computed(() => plans.value.find((plan) => plan.id === activePlanId.value));

const orderedStrategies = computed(() => {
  const order: StrategyId[] = [
    "freshness", "proximity", "emd", "entropy", "bayesian",
    "markov100", "svc", "tbl", "randomness",
  ];
  return order
    .map((id) => strategyById.value.get(id))
    .filter((strategy): strategy is StrategyPrediction => strategy !== undefined);
});

const focusedPredictions = computed(() => {
  const number = focusedNumber.value;
  return new Map(
    orderedStrategies.value.map((strategy) => [
      strategy.id,
      number === null
        ? null
        : strategy.numbers.find((prediction) => prediction.number === number) ?? null,
    ]),
  );
});

const randomnessRows = computed(() => strategyById.value.get("randomness")?.numbers ?? []);
const agreementScore = computed(() => {
  if (selectedNumbers.value.length === 0) return 0;
  const rankByNumber = new Map(randomnessRows.value.map((row) => [row.number, row.rank]));
  const strength = selectedNumbers.value.reduce(
    (total, number) => total + (49 - (rankByNumber.get(number) ?? 49)) / 48,
    0,
  ) / selectedNumbers.value.length;
  const top = new Set(strategyById.value.get("randomness")?.topNumbers ?? []);
  const overlap = selectedNumbers.value.filter((number) => top.has(number)).length / 6;
  const completeness = selectedNumbers.value.length / 6;
  return Math.round((strength * 0.55 + overlap * 0.35 + completeness * 0.1) * 100);
});
const agreementLabel = computed(() => {
  if (selectedNumbers.value.length === 0) return "Select numbers";
  if (agreementScore.value >= 80) return "Very strong";
  if (agreementScore.value >= 60) return "Strong";
  if (agreementScore.value >= 40) return "Moderate";
  return "Low";
});

const entropyStatus = computed(() => {
  if (selectedNumbers.value.length !== 6) return `Need ${6 - selectedNumbers.value.length}`;
  const ordered = [...selectedNumbers.value].sort((left, right) => left - right);
  const gaps = ordered.slice(1).map((number, index) => number - ordered[index]);
  gaps.push(49 + ordered[0] - ordered.at(-1)!);
  const entropy = -gaps.reduce((total, gap) => {
    const probability = gap / 49;
    return total + probability * Math.log2(probability);
  }, 0) / Math.log2(6);
  return entropy >= 0.9 ? "Correlated" : entropy >= 0.78 ? "Balanced" : "Clustered";
});

const edgeByPair = computed(() => {
  const entries = (dialogData.value?.possibleDraw.relationshipEdges ?? []).map(
    (edge) => [`${edge.left}-${edge.right}`, edge] as const,
  );
  return new Map<string, RelationshipEdge>(entries);
});
const relationSeeds = computed(() =>
  selectedNumbers.value.length > 0
    ? selectedNumbers.value
    : focusedNumber.value === null ? [] : [focusedNumber.value],
);
const relatedSuggestions = computed<RelatedSuggestion[]>(() => {
  if (relationSeeds.value.length === 0 || selectedNumbers.value.length >= 6) return [];
  return Array.from({ length: 49 }, (_value, index) => index + 1)
    .filter(
      (candidate) =>
        !selectedSet.value.has(candidate) &&
        !droppedSet.value.has(candidate) &&
        !relationSeeds.value.includes(candidate),
    )
    .map((candidate) => {
      const edges = relationSeeds.value
        .map((seed) => edgeByPair.value.get(pairKey(seed, candidate)))
        .filter((edge): edge is RelationshipEdge => edge !== undefined);
      const lift = edges.length
        ? edges.reduce((total, edge) => total + edge.lift, 0) / edges.length
        : 0;
      const hits = edges.reduce((total, edge) => total + edge.count, 0);
      const residual = edges.reduce((total, edge) => total + edge.residual, 0);
      return { number: candidate, lift, hits, score: lift * 100 + hits + residual * 8 };
    })
    .sort((left, right) => right.score - left.score || right.hits - left.hits || left.number - right.number)
    .slice(0, 5);
});
const strongestRelated = computed(() => relatedSuggestions.value[0] ?? null);

function newPlan(name = `Draw ${plans.value.length + 1}`): DrawPlan {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    name,
    selected: [],
    dropped: [],
    uncertain: [],
  };
}

function savePlans(): void {
  const active = activePlan.value;
  if (active) {
    active.selected = [...selectedNumbers.value];
    active.dropped = [...droppedNumbers.value];
    active.uncertain = [...uncertainNumbers.value];
  }
  window.localStorage.setItem(
    storageKey,
    JSON.stringify({ activePlanId: activePlanId.value, plans: plans.value }),
  );
}

function loadPlan(plan: DrawPlan): void {
  activePlanId.value = plan.id;
  selectedNumbers.value = [...plan.selected];
  droppedNumbers.value = [...plan.dropped];
  uncertainNumbers.value = [...plan.uncertain];
  focusedNumber.value = null;
}

function switchPlan(planId: string): void {
  savePlans();
  const plan = plans.value.find((candidate) => candidate.id === planId);
  if (plan) loadPlan(plan);
}

function createPlan(): void {
  savePlans();
  const plan = newPlan();
  plans.value = [...plans.value, plan];
  loadPlan(plan);
  savePlans();
}

function deletePlan(): void {
  if (plans.value.length === 1) {
    resetPlan();
    return;
  }
  const index = plans.value.findIndex((plan) => plan.id === activePlanId.value);
  plans.value = plans.value.filter((plan) => plan.id !== activePlanId.value);
  loadPlan(plans.value[Math.min(Math.max(index, 0), plans.value.length - 1)]);
  savePlans();
}

function clearClickTimer(): void {
  if (clickTimer !== null) {
    clearTimeout(clickTimer);
    clickTimer = null;
  }
}

function toggleSelected(number: number): void {
  if (droppedSet.value.has(number)) return;
  if (selectedSet.value.has(number)) {
    selectedNumbers.value = selectedNumbers.value.filter((item) => item !== number);
  } else if (selectedNumbers.value.length < 6) {
    selectedNumbers.value = [...selectedNumbers.value, number].sort((left, right) => left - right);
    uncertainNumbers.value = uncertainNumbers.value.filter((item) => item !== number);
  }
}

function handleClick(event: MouseEvent, number: number): void {
  clearClickTimer();
  focusedNumber.value = number;
  if (event.ctrlKey && event.altKey) return;
  clickTimer = setTimeout(() => {
    toggleSelected(number);
    clickTimer = null;
  }, 220);
}

function handleDoubleClick(event: MouseEvent, number: number): void {
  clearClickTimer();
  focusedNumber.value = number;
  if (event.ctrlKey) {
    if (droppedSet.value.has(number)) return;
    uncertainNumbers.value = uncertainSet.value.has(number)
      ? uncertainNumbers.value.filter((item) => item !== number)
      : [...uncertainNumbers.value, number].sort((left, right) => left - right);
    return;
  }
  if (droppedSet.value.has(number)) {
    droppedNumbers.value = droppedNumbers.value.filter((item) => item !== number);
  } else {
    droppedNumbers.value = [...droppedNumbers.value, number].sort((left, right) => left - right);
    selectedNumbers.value = selectedNumbers.value.filter((item) => item !== number);
    uncertainNumbers.value = uncertainNumbers.value.filter((item) => item !== number);
  }
}

function addRelated(number: number): void {
  clearClickTimer();
  focusedNumber.value = number;
  toggleSelected(number);
}

function resetPlan(): void {
  clearClickTimer();
  selectedNumbers.value = [];
  droppedNumbers.value = [];
  uncertainNumbers.value = [];
}

function stepLastSeen(direction: -1 | 1): void {
  if (!lastSeenRows.value.length) return;
  showLastSeen.value = true;
  lastSeenIndex.value =
    (lastSeenIndex.value + direction + lastSeenRows.value.length) % lastSeenRows.value.length;
  focusedNumber.value = highlightedLastSeen.value?.number ?? null;
}

function pairKey(left: number, right: number): string {
  return `${Math.min(left, right)}-${Math.max(left, right)}`;
}

function rankWidth(prediction: StrategyNumberPrediction | null | undefined): string {
  return `${prediction ? Math.max(2, ((50 - prediction.rank) / 49) * 100) : 0}%`;
}

function cardColor(id: string): string {
  return {
    freshness: "#f58a59", proximity: "#efb23e", emd: "#d9a531",
    entropy: "#c95d42", bayesian: "#f3b94e", markov100: "#f3b94e",
    svc: "#9567e8", tbl: "#1695a8", randomness: "#3264ad",
  }[id] ?? "#6e8195";
}

function compactDetail(detail: string): string {
  return detail
    .replace("Average entropy", "Avg")
    .replace("High-entropy share", "High")
    .replace("Posterior probability", "Prob")
    .replace("Average distance", "EMD")
    .replace("Support draws", "Hits")
    .replace("Hit probability", "Hit")
    .replace("Lifetime frequency", "Life");
}

function acceptData(data: CombinedPredictionDialogData): void {
  dialogData.value = data;
  if (focusedNumber.value === null) {
    focusedNumber.value = data.possibleDraw.lastSeenRows[0]?.number ?? 1;
  }
}

watch(
  [selectedNumbers, droppedNumbers, uncertainNumbers],
  () => {
    if (plans.value.length) savePlans();
  },
  { deep: true },
);

onMounted(async () => {
  document.title = "Possible Draw — Rand AI";
  try {
    const stored = JSON.parse(window.localStorage.getItem(storageKey) ?? "null") as
      | { activePlanId: string; plans: DrawPlan[] }
      | null;
    plans.value = stored?.plans?.length ? stored.plans : [newPlan("Draw 1")];
    loadPlan(plans.value.find((plan) => plan.id === stored?.activePlanId) ?? plans.value[0]);
  } catch {
    plans.value = [newPlan("Draw 1")];
    loadPlan(plans.value[0]);
  }
  if (!window.randAiDesktop) {
    errorMessage.value = "Possible Draw is available inside the Electron application.";
    return;
  }
  unsubscribeData = window.randAiDesktop.onCombinedPredictionData(acceptData);
  try {
    const data = await window.randAiDesktop.getCombinedPredictionData();
    if (data) acceptData(data);
    else errorMessage.value = "Analyze a dataset before opening Possible Draw.";
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
});

onBeforeUnmount(() => {
  clearClickTimer();
  unsubscribeData?.();
});
</script>

<template>
  <main class="possible-draw-dialog-shell">
    <section v-if="latestSuite && dialogData" class="possible-draw-window">
      <header class="possible-draw-toolbar">
        <label class="possible-plan-select">
          <span>Plan</span>
          <select :value="activePlanId" @change="switchPlan(($event.target as HTMLSelectElement).value)">
            <option v-for="plan in plans" :key="plan.id" :value="plan.id">{{ plan.name }}</option>
          </select>
        </label>
        <button type="button" @click="createPlan">New Draw</button>
        <button type="button" @click="deletePlan">Delete Draw</button>
        <div class="status-pill agreement">
          <strong>Model Agreement {{ agreementScore }}%</strong>
          <span>{{ agreementLabel }}</span>
        </div>
        <div class="status-pill entropy">
          <strong>Entropy {{ entropyStatus }}</strong>
        </div>
        <label class="possible-toggle">
          <input v-model="showLastDraw" type="checkbox">
          <span>Last draw rings</span>
        </label>
        <label class="possible-toggle">
          <input v-model="showLastSeen" type="checkbox">
          <span>Last seen</span>
        </label>
        <button type="button" class="square-button" @click="stepLastSeen(-1)">&lt;</button>
        <div class="last-seen-pill">
          <strong>{{ highlightedLastSeen?.number ?? "—" }}</strong>
          <span>Gap {{ highlightedLastSeen?.gap ?? "—" }}</span>
        </div>
        <button type="button" class="square-button" @click="stepLastSeen(1)">&gt;</button>
        <button type="button" @click="resetPlan">Reset</button>
      </header>

      <div class="possible-draw-layout">
        <section class="possible-number-board">
          <div class="draw-grid" role="grid" aria-label="Possible draw numbers">
            <button
              v-for="number in 49"
              :key="number"
              type="button"
              class="draw-cell"
              :class="{
                available: !droppedSet.has(number),
                selected: selectedSet.has(number),
                dropped: droppedSet.has(number),
                uncertain: uncertainSet.has(number),
                focused: focusedNumber === number,
                lastDraw: showLastDraw && lastDrawSet.has(number),
                lastSeen: showLastSeen && highlightedLastSeen?.number === number,
              }"
              :aria-pressed="selectedSet.has(number)"
              :title="`Number ${number}: click to select, double-click to exclude, Ctrl+double-click for uncertain`"
              @click="handleClick($event, number)"
              @dblclick="handleDoubleClick($event, number)"
            >
              {{ number }}
            </button>
          </div>
          <p class="possible-help">
            Click to select up to six · Double-click to exclude · Ctrl+double-click to mark uncertain
          </p>
        </section>

        <section class="possible-analysis-panel">
          <div class="prediction-meters-panel">
            <article
              v-for="strategy in orderedStrategies"
              :key="strategy.id"
              class="prediction-meter-card"
              :style="{ '--meter-color': cardColor(strategy.id) }"
            >
              <h2>{{ strategy.name }}</h2>
              <div class="meter-scale"><span>1</span><span>13</span><span>25</span><span>37</span><span>49</span></div>
              <div class="meter-track">
                <span :style="{ width: rankWidth(focusedPredictions.get(strategy.id)) }"></span>
              </div>
              <div v-if="focusedPredictions.get(strategy.id)" class="meter-details">
                <strong>Rank {{ focusedPredictions.get(strategy.id)!.rank }}</strong>
                <span
                  v-for="detail in focusedPredictions.get(strategy.id)!.details.slice(0, 3)"
                  :key="detail"
                  :title="detail"
                >{{ compactDetail(detail) }}</span>
                <span>Score {{ (focusedPredictions.get(strategy.id)!.score * 100).toFixed(1) }}</span>
              </div>
              <div v-else class="meter-details"><strong>Select a number</strong></div>
            </article>
          </div>

          <article class="related-number-card">
            <div class="related-summary">
              <span>Next Related</span>
              <strong>{{ strongestRelated?.number ?? "—" }}</strong>
              <b>Avg lift x{{ (strongestRelated?.lift ?? 0).toFixed(2) }}</b>
              <small>Pair hits {{ strongestRelated?.hits ?? 0 }}</small>
              <small>Related to {{ relationSeeds.length }} selected</small>
            </div>
            <div class="relationship-visual">
              <span class="seed-ball">{{ relationSeeds[0] ?? "—" }}</span>
              <i><span :style="{ width: `${Math.min((strongestRelated?.lift ?? 0) * 50, 100)}%` }"></span></i>
              <button
                type="button"
                class="related-ball"
                :disabled="!strongestRelated"
                @click="strongestRelated && addRelated(strongestRelated.number)"
              >
                {{ strongestRelated?.number ?? "—" }}
              </button>
            </div>
            <div class="related-list">
              <button
                v-for="suggestion in relatedSuggestions"
                :key="suggestion.number"
                type="button"
                @click="addRelated(suggestion.number)"
              >
                <strong>{{ suggestion.number }}</strong>
                <span>vs {{ relationSeeds.length }} selected</span>
                <i><span :style="{ width: `${Math.min(suggestion.lift * 50, 100)}%` }"></span></i>
                <small>x{{ suggestion.lift.toFixed(2) }}</small>
              </button>
            </div>
          </article>
        </section>
      </div>
    </section>

    <section v-else class="dialog-empty-state">
      <strong>Possible Draw unavailable</strong>
      <p>{{ errorMessage || "Loading Python-calculated predictions…" }}</p>
    </section>
  </main>
</template>
