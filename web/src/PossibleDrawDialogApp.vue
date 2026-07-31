<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type {
  CombinedPredictionDialogData,
  PossibleDrawNumberRequest,
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

interface RecalculatedPrediction extends StrategyNumberPrediction {
  originalRank: number;
}

interface WorkflowCandidate {
  number: number;
  consensus: number;
  topSixSupport: number;
  strategyCount: number;
  relationshipLift: number;
  combinedScore: number;
}

const props = defineProps<{
  dialogData: CombinedPredictionDialogData;
  embedded?: boolean;
  numberRequest?: (PossibleDrawNumberRequest & { token: number }) | null;
}>();

const storageKey = "rand-ai.possible-draw.plans.v2";
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

const latestSuite = computed(() => props.dialogData.predictionSuites.at(-1) ?? null);
const strategies = computed(() => latestSuite.value?.strategies ?? []);
const strategyById = computed(
  () => new Map(strategies.value.map((strategy) => [strategy.id, strategy])),
);
const selectedSet = computed(() => new Set(selectedNumbers.value));
const droppedSet = computed(() => new Set(droppedNumbers.value));
const uncertainSet = computed(() => new Set(uncertainNumbers.value));
const lastDrawSet = computed(
  () => new Set(props.dialogData.possibleDraw.lastDrawNumbers),
);
const lastSeenRows = computed(() => props.dialogData.possibleDraw.lastSeenRows);
const highlightedLastSeen = computed(() => lastSeenRows.value[lastSeenIndex.value] ?? null);
const activePlan = computed(() => plans.value.find((plan) => plan.id === activePlanId.value));

const orderedStrategies = computed(() => {
  const fallbackOrder: StrategyId[] = [
    "freshness", "proximity", "emd", "chi_square", "entropy", "markov100",
    "mkfr", "mksp", "mknp", "bayesian", "predictive_grid", "co_occurrence",
    "doublet_triplet_markov", "mixed", "svc", "tbl",
    "cis", "fresh_random", "randomness",
    "residual_coverage",
    "chained",
  ];
  const fallbackRank = new Map(fallbackOrder.map((id, index) => [id, index]));
  return [...strategies.value].sort((left, right) => {
    const leftEfficacy = left.efficacy;
    const rightEfficacy = right.efficacy;
    const leftHasHistory = (leftEfficacy?.evaluatedDraws ?? 0) > 0;
    const rightHasHistory = (rightEfficacy?.evaluatedDraws ?? 0) > 0;

    if (leftHasHistory !== rightHasHistory) return rightHasHistory ? 1 : -1;

    const averageHitDifference =
      (rightEfficacy?.averageHitsPerDraw ?? 0) -
      (leftEfficacy?.averageHitsPerDraw ?? 0);
    if (averageHitDifference !== 0) return averageHitDifference;

    const randomDifference =
      (rightEfficacy?.hitDifference ?? 0) - (leftEfficacy?.hitDifference ?? 0);
    if (randomDifference !== 0) return randomDifference;

    const totalHitDifference =
      (rightEfficacy?.strategyHits ?? 0) - (leftEfficacy?.strategyHits ?? 0);
    if (totalHitDifference !== 0) return totalHitDifference;

    return (
      (fallbackRank.get(left.id) ?? fallbackOrder.length) -
      (fallbackRank.get(right.id) ?? fallbackOrder.length)
    );
  });
});

const recalculatedPredictions = computed(() => {
  const predictions = new Map<StrategyId, RecalculatedPrediction[]>();
  for (const strategy of orderedStrategies.value) {
    predictions.set(
      strategy.id,
      strategy.numbers
        .filter(
          (prediction) =>
            !selectedSet.value.has(prediction.number) &&
            !droppedSet.value.has(prediction.number),
        )
        .sort((left, right) => left.rank - right.rank)
        .map((prediction, index) => ({
          ...prediction,
          originalRank: prediction.rank,
          rank: index + 1,
        })),
    );
  }
  return predictions;
});

const focusedPredictions = computed(() => {
  const number = focusedNumber.value;
  return new Map(
    orderedStrategies.value.map((strategy) => [
      strategy.id,
      number === null
        ? null
        : recalculatedPredictions.value
          .get(strategy.id)
          ?.find((prediction) => prediction.number === number) ?? null,
    ]),
  );
});

const remainingNumberCount = computed(
  () => 49 - selectedSet.value.size - droppedSet.value.size,
);
const workflowComplete = computed(() => selectedNumbers.value.length === 6);
const currentWorkflowStep = computed(() => Math.min(selectedNumbers.value.length + 1, 6));
const workflowCandidates = computed<WorkflowCandidate[]>(() => {
  const candidates: WorkflowCandidate[] = [];
  const strategyCount = orderedStrategies.value.length;
  const possibleNumbers = Array.from({ length: 49 }, (_value, index) => index + 1)
    .filter(
      (number) => !selectedSet.value.has(number) && !droppedSet.value.has(number),
    );

  for (const number of possibleNumbers) {
    let weightedStrength = 0;
    let totalWeight = 0;
    let topSixSupport = 0;

    for (const strategy of orderedStrategies.value) {
      const rows = recalculatedPredictions.value.get(strategy.id) ?? [];
      const prediction = rows.find((row) => row.number === number);
      if (!prediction) continue;
      const weight = Math.max(strategy.efficacy?.averageHitsPerDraw ?? 0.1, 0.05);
      const rankStrength = rows.length <= 1
        ? 1
        : (rows.length - prediction.rank) / (rows.length - 1);
      weightedStrength += rankStrength * weight;
      totalWeight += weight;
      if (prediction.rank <= Math.min(6, rows.length)) topSixSupport += 1;
    }

    const relationshipEdges = selectedNumbers.value
      .map((selected) => edgeByPair.value.get(pairKey(selected, number)))
      .filter((edge): edge is RelationshipEdge => edge !== undefined);
    const relationshipLift = relationshipEdges.length
      ? relationshipEdges.reduce((total, edge) => total + edge.lift, 0) /
        relationshipEdges.length
      : 0;
    const consensus = totalWeight > 0 ? weightedStrength / totalWeight : 0;
    const supportRate = strategyCount > 0 ? topSixSupport / strategyCount : 0;
    const relationshipStrength = selectedNumbers.value.length
      ? Math.min(Math.max(relationshipLift / 2, 0), 1)
      : 0.5;

    candidates.push({
      number,
      consensus,
      topSixSupport,
      strategyCount,
      relationshipLift,
      combinedScore:
        consensus * 0.72 + supportRate * 0.18 + relationshipStrength * 0.1,
    });
  }

  return candidates.sort(
    (left, right) =>
      right.combinedScore - left.combinedScore ||
      right.topSixSupport - left.topSixSupport ||
      left.number - right.number,
  );
});
const topWorkflowCandidates = computed(() => workflowCandidates.value.slice(0, 8));
const topWorkflowCandidateSet = computed(
  () => new Set(topWorkflowCandidates.value.map((candidate) => candidate.number)),
);
const strategyNextChoices = computed(() =>
  orderedStrategies.value.map((strategy) => ({
    id: strategy.id,
    name: strategyFullName(strategy),
    prediction: recalculatedPredictions.value.get(strategy.id)?.[0] ?? null,
  })),
);

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
  const entries = props.dialogData.possibleDraw.relationshipEdges.map(
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
    selectedNumbers.value = [...selectedNumbers.value, number];
    uncertainNumbers.value = uncertainNumbers.value.filter((item) => item !== number);
  } else {
    void window.randAiDesktop?.showForSureLimitError(number);
  }
}

function applyPredictionNumber(request: PossibleDrawNumberRequest): void {
  const number = request.number;
  if (!Number.isInteger(number) || number < 1 || number > 49) return;
  clearClickTimer();
  focusedNumber.value = number;
  droppedNumbers.value = droppedNumbers.value.filter((item) => item !== number);

  if (request.state === "possible") {
    selectedNumbers.value = selectedNumbers.value.filter((item) => item !== number);
    if (!uncertainSet.value.has(number)) {
      uncertainNumbers.value = [...uncertainNumbers.value, number].sort(
        (left, right) => left - right,
      );
    }
    return;
  }

  if (selectedSet.value.has(number)) return;
  if (selectedNumbers.value.length >= 6) {
    void window.randAiDesktop?.showForSureLimitError(number);
    return;
  }
  uncertainNumbers.value = uncertainNumbers.value.filter((item) => item !== number);
  selectedNumbers.value = [...selectedNumbers.value, number];
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

function chooseWorkflowNumber(number: number): void {
  clearClickTimer();
  focusedNumber.value = number;
  if (!selectedSet.value.has(number)) toggleSelected(number);
}

function removeWorkflowNumber(number: number): void {
  clearClickTimer();
  focusedNumber.value = number;
  selectedNumbers.value = selectedNumbers.value.filter((item) => item !== number);
}

function undoLastWorkflowNumber(): void {
  const number = selectedNumbers.value.at(-1);
  if (number === undefined) return;
  removeWorkflowNumber(number);
}

function selectionStep(number: number): number {
  return selectedNumbers.value.indexOf(number) + 1;
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

function strategyMeterTitle(strategy: StrategyPrediction): string {
  const prediction = focusedPredictions.value.get(strategy.id);
  const effectiveness = strategy.efficacy && strategy.efficacy.evaluatedDraws > 0
    ? `${strategy.efficacy.averageHitsPerDraw.toFixed(3)} historical hits per draw`
    : "historical effectiveness unavailable";
  if (!prediction) {
    return `${strategyFullName(strategy)} · ${effectiveness}`;
  }
  return `${strategyFullName(strategy)} · rank ${prediction.rank} · ${effectiveness}`;
}

function strategyFullName(strategy: StrategyPrediction): string {
  return {
    freshness: "Freshness",
    proximity: "Proximity",
    emd: "Earth Mover's Distance",
    chi_square: "Chi-Square",
    entropy: "Entropy",
    markov100: "Markov 100",
    mkfr: "Markov Frequency",
    mksp: "Markov Spatial",
    mknp: "Markov Normalized Positions",
    bayesian: "Bayesian",
    predictive_grid: "Predictive Grid",
    co_occurrence: "Co-occurrence",
    doublet_triplet_markov: "Doublet & Triplet Markov",
    mixed: "Mixed Ensemble",
    svc: "Support Vector Classifier",
    tbl: "Trend Baseline",
    cis: "Conditional Independence Score",
    fresh_random: "Fresh Random",
    randomness: "Randomness Ensemble",
    residual_coverage: "Residual Coverage",
    chained: "Chained Strategy",
  }[strategy.id] ?? strategy.name;
}

function cardColor(id: string): string {
  return {
    freshness: "#f58a59", proximity: "#efb23e", emd: "#d9a531",
    entropy: "#c95d42", markov100: "#f3b94e", mkfr: "#1f8f75",
    mksp: "#517aa3",
    mknp: "#2f7f9f",
    chi_square: "#6256c7", bayesian: "#d477b8",
    predictive_grid: "#008ca8", co_occurrence: "#4f7f3f",
    doublet_triplet_markov: "#7c3aed", mixed: "#dd6b20",
    svc: "#9567e8", tbl: "#1695a8", cis: "#b12f67",
    fresh_random: "#7b56c2", randomness: "#3264ad",
    residual_coverage: "#0f766e",
    chained: "#9a3412",
  }[id] ?? "#6e8195";
}

function acceptData(data: CombinedPredictionDialogData): void {
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

watch(
  () => props.dialogData,
  (data) => acceptData(data),
  { immediate: true },
);

watch(
  () => props.numberRequest?.token,
  () => {
    if (props.numberRequest) applyPredictionNumber(props.numberRequest);
  },
);

onMounted(() => {
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
  if (props.numberRequest) applyPredictionNumber(props.numberRequest);
});

onBeforeUnmount(() => {
  clearClickTimer();
});
</script>

<template>
  <main
    class="possible-draw-dialog-shell"
    :class="{ 'embedded-possible-draw': embedded }"
  >
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

      <section class="guided-draw-workflow" aria-labelledby="guided-draw-title">
        <header class="guided-workflow-header">
          <div>
            <span class="guided-workflow-eyebrow">Guided prediction workflow</span>
            <h1 id="guided-draw-title">
              {{ workflowComplete ? "Predicted draw complete" : `Choose number ${currentWorkflowStep} of 6` }}
            </h1>
            <p>
              Every chosen number is eliminated from all strategy lists. Remaining ranks,
              strategy leaders, and the combined recommendation are recalculated after each pick.
            </p>
          </div>
          <div class="guided-workflow-progress" :class="{ complete: workflowComplete }">
            <strong>{{ selectedNumbers.length }}/6</strong>
            <span>{{ workflowComplete ? "Complete" : `${remainingNumberCount} candidates remain` }}</span>
          </div>
        </header>

        <ol class="guided-pick-steps" aria-label="Six-number prediction sequence">
          <li
            v-for="step in 6"
            :key="step"
            :class="{
              complete: selectedNumbers[step - 1] !== undefined,
              active: !workflowComplete && step === currentWorkflowStep,
            }"
          >
            <button
              v-if="selectedNumbers[step - 1] !== undefined"
              type="button"
              :title="`Remove pick ${step} and recalculate`"
              @click="removeWorkflowNumber(selectedNumbers[step - 1])"
            >
              <span>Pick {{ step }}</span>
              <strong>{{ selectedNumbers[step - 1] }}</strong>
              <small>Remove</small>
            </button>
            <div v-else>
              <span>Pick {{ step }}</span>
              <strong>—</strong>
              <small>{{ step === currentWorkflowStep ? "Choose next" : "Waiting" }}</small>
            </div>
          </li>
        </ol>

        <div v-if="!workflowComplete" class="guided-recommendations">
          <div class="guided-section-heading">
            <div>
              <strong>Recalculated next choices</strong>
              <span>Ranked using all {{ orderedStrategies.length }} strategies and relationships to your locked picks.</span>
            </div>
            <button
              type="button"
              :disabled="selectedNumbers.length === 0"
              @click="undoLastWorkflowNumber"
            >
              Undo last pick
            </button>
          </div>
          <div class="guided-candidate-grid">
            <button
              v-for="(candidate, index) in topWorkflowCandidates"
              :key="candidate.number"
              type="button"
              @click="chooseWorkflowNumber(candidate.number)"
            >
              <span class="candidate-position">#{{ index + 1 }}</span>
              <strong>{{ candidate.number }}</strong>
              <span class="candidate-consensus">
                {{ Math.round(candidate.consensus * 100) }}% rank strength
              </span>
              <small>
                Top 6 in {{ candidate.topSixSupport }}/{{ candidate.strategyCount }} strategies
              </small>
              <small v-if="selectedNumbers.length">
                Relationship lift ×{{ candidate.relationshipLift.toFixed(2) }}
              </small>
            </button>
          </div>
        </div>

        <div v-else class="guided-complete-summary">
          <div>
            <span>Final predicted draw, in selection order</span>
            <strong>{{ selectedNumbers.join(" · ") }}</strong>
          </div>
          <button type="button" @click="undoLastWorkflowNumber">Reopen last pick</button>
        </div>

        <details class="guided-strategy-leaders">
          <summary>All recalculated strategy leaders</summary>
          <div>
            <article v-for="choice in strategyNextChoices" :key="choice.id">
              <span>{{ choice.name }}</span>
              <strong>{{ choice.prediction?.number ?? "—" }}</strong>
              <small v-if="choice.prediction">
                Now #1 · originally #{{ choice.prediction.originalRank }}
              </small>
              <small v-else>No remaining candidate</small>
            </article>
          </div>
        </details>
      </section>

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
                recommended: topWorkflowCandidateSet.has(number) && !workflowComplete,
              }"
              :aria-pressed="selectedSet.has(number)"
              :title="selectedSet.has(number)
                ? `Number ${number}: workflow pick ${selectionStep(number)}`
                : `Number ${number}: click to choose next, double-click to exclude, Ctrl+double-click for Possible`"
              @click="handleClick($event, number)"
              @dblclick="handleDoubleClick($event, number)"
            >
              {{ number }}
              <small v-if="selectedSet.has(number)" class="selection-order">
                {{ selectionStep(number) }}
              </small>
            </button>
          </div>
          <p class="possible-help">
            For Sure: click, maximum six · Exclude: double-click · Possible: Ctrl+double-click, no six-number limit
          </p>
        </section>

        <section class="possible-analysis-panel">
          <div class="prediction-meters-panel">
            <article
              v-for="strategy in orderedStrategies"
              :key="strategy.id"
              class="prediction-meter-card"
              :style="{ '--meter-color': cardColor(strategy.id) }"
              :title="strategyMeterTitle(strategy)"
            >
              <h2>{{ strategyFullName(strategy) }}</h2>
              <div
                class="meter-track"
                role="meter"
                aria-valuemin="1"
                aria-valuemax="49"
                :aria-valuenow="focusedPredictions.get(strategy.id)?.rank"
                :aria-label="strategyMeterTitle(strategy)"
              >
                <span :style="{ width: rankWidth(focusedPredictions.get(strategy.id)) }"></span>
              </div>
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
