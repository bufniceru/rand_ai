<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { strategyColor } from "./lib/strategyColors";
import {
  activePossibleDrawPlan,
  activePossibleDrawPlanId,
  createPossibleDrawPlan,
  cyclePossibleDrawNumberState,
  deletePossibleDrawPlan,
  getPossibleDrawNumberState,
  possibleDrawPlans,
  replacePossibleDrawNumbers,
  resetPossibleDrawPlan,
  selectPossibleDrawPlan,
  setPossibleDrawNumberState,
  togglePossibleDrawExcluded,
} from "./lib/possibleDrawPlans";
import type {
  CombinedPredictionDialogData,
  RelationshipEdge,
  StrategyId,
  StrategyNumberPrediction,
  StrategyPrediction,
} from "./types";

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
}>();

const plans = possibleDrawPlans;
const activePlanId = activePossibleDrawPlanId;
const selectedNumbers = computed({
  get: () => activePossibleDrawPlan.value?.fixedNumbers ?? [],
  set: (numbers: number[]) => replacePossibleDrawNumbers("fixedNumbers", numbers),
});
const droppedNumbers = computed({
  get: () => activePossibleDrawPlan.value?.excludedNumbers ?? [],
  set: (numbers: number[]) => replacePossibleDrawNumbers("excludedNumbers", numbers),
});
const uncertainNumbers = computed({
  get: () => activePossibleDrawPlan.value?.candidateNumbers ?? [],
  set: (numbers: number[]) => replacePossibleDrawNumbers("candidateNumbers", numbers),
});
const focusedNumber = ref<number | null>(null);
const showLastDraw = ref(false);
const showLastSeen = ref(true);
const lastSeenIndex = ref(0);
const errorMessage = ref("");

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

const orderedStrategies = computed(() => {
  const fallbackOrder: StrategyId[] = [
    "freshness", "proximity", "emd", "chi_square", "categorical_chi_square", "entropy", "markov100", "mkgsv",
    "mkfr", "mksp", "mknp", "mkrd", "bayesian", "predictive_grid", "co_occurrence",
    "doublet_triplet_markov", "mixed", "svc", "tbl",
    "cis", "fresh_random", "randomness",
    "decision_tree_selector",
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
      Number(uncertainSet.value.has(right.number)) -
        Number(uncertainSet.value.has(left.number)) ||
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

function switchPlan(planId: string): void {
  selectPossibleDrawPlan(planId);
  focusedNumber.value = null;
}

function createPlan(): void {
  createPossibleDrawPlan();
  focusedNumber.value = null;
}

function deletePlan(): void {
  deletePossibleDrawPlan();
  focusedNumber.value = null;
}

function toggleSelected(number: number): void {
  const result = setPossibleDrawNumberState(
    number,
    selectedSet.value.has(number) ? "neutral" : "fixed",
  );
  applyStateResult(number, result);
}

function handleClick(event: MouseEvent, number: number): void {
  focusedNumber.value = number;
  applyStateResult(
    number,
    event.altKey ? togglePossibleDrawExcluded(number) : cyclePossibleDrawNumberState(number),
  );
}

function applyStateResult(
  number: number,
  result: { ok: boolean; message?: string },
): void {
  errorMessage.value = result.message ?? "";
  if (!result.ok) {
    void window.randAiDesktop?.showForSureLimitError(number);
  }
}

function handleNumberKeydown(event: KeyboardEvent, number: number): void {
  const key = event.key.toLowerCase();
  let result: { ok: boolean; message?: string } | null = null;
  if (key === "enter" || key === " ") result = cyclePossibleDrawNumberState(number);
  if (key === "c") result = setPossibleDrawNumberState(number, "candidate");
  if (key === "f") result = setPossibleDrawNumberState(number, "fixed");
  if (key === "x") result = togglePossibleDrawExcluded(number);
  if (key === "delete" || key === "backspace") {
    result = setPossibleDrawNumberState(number, "neutral");
  }
  if (!result) return;
  event.preventDefault();
  focusedNumber.value = number;
  applyStateResult(number, result);
}

function addRelated(number: number): void {
  focusedNumber.value = number;
  toggleSelected(number);
}

function chooseWorkflowNumber(number: number): void {
  focusedNumber.value = number;
  if (!selectedSet.value.has(number)) toggleSelected(number);
}

function removeWorkflowNumber(number: number): void {
  focusedNumber.value = number;
  setPossibleDrawNumberState(number, "neutral");
}

function undoLastWorkflowNumber(): void {
  const number = selectedNumbers.value.at(-1);
  if (number === undefined) return;
  removeWorkflowNumber(number);
}

function selectionStep(number: number): number {
  return selectedNumbers.value.indexOf(number) + 1;
}

function numberState(number: number) {
  return getPossibleDrawNumberState(number);
}

function resetPlan(): void {
  resetPossibleDrawPlan();
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
    recurrence_dynamics: "Recurrence Dynamics (Experimental)",
    chi_square: "Chi-Square",
    categorical_chi_square: "Categorical Chi-square",
    entropy: "Entropy",
    markov100: "Markov 100",
    mkgsv: "Markov Gap-Space Vector (Experimental)",
    mkfr: "Markov Frequency",
    mksp: "Markov Spatial",
    mknp: "Markov Normalized Positions",
    mkrd: "Markov Relative Dispersion",
    bayesian: "Bayesian",
    predictive_grid: "Predictive Grid",
    co_occurrence: "Co-occurrence",
    doublet_triplet_markov: "Doublet & Triplet Markov",
    mixed: "Mixed Ensemble",
    svc: "Support Vector Classifier",
    tbl: "Trend Baseline",
    sklearn_svm: "Scikit Online SVM",
    lag_logistic: "Lagged Logistic",
    sparse_neural_ticket: "Sparse Neural Ticket (Experimental)",
    cis: "Conditional Independence Score",
    decision_tree_selector: "Decision Tree Selector",
    fresh_random: "Fresh Random",
    randomness: "Randomness Ensemble",
    residual_coverage: "Residual Coverage",
    chained: "Chained Strategy",
    border_group_statistical: "Border Group Statistical",
    border_group_markov: "Border Group Markov",
    border_group_bayesian: "Border Group Bayesian",
    border_group_ml: "Border Group ML",
    border_group_hybrid: "Border Group Hybrid",
  }[strategy.id] ?? strategy.name;
}

function acceptData(data: CombinedPredictionDialogData): void {
  if (focusedNumber.value === null) {
    focusedNumber.value = data.possibleDraw.lastSeenRows[0]?.number ?? 1;
  }
}

watch(
  () => props.dialogData,
  (data) => acceptData(data),
  { immediate: true },
);

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
                fixed: selectedSet.has(number),
                excluded: droppedSet.has(number),
                candidate: uncertainSet.has(number),
                focused: focusedNumber === number,
                lastDraw: showLastDraw && lastDrawSet.has(number),
                lastSeen: showLastSeen && highlightedLastSeen?.number === number,
                recommended: topWorkflowCandidateSet.has(number) && !workflowComplete,
              }"
              :aria-label="`Number ${number}; ${numberState(number)} Possible Draw state`"
              :aria-pressed="numberState(number) !== 'neutral'"
              :title="`Number ${number}: ${numberState(number)}. Click cycles Neutral, Candidate, and Fixed; Alt+click or X toggles Excluded.`"
              @click="handleClick($event, number)"
              @keydown="handleNumberKeydown($event, number)"
            >
              {{ number }}
              <small v-if="selectedSet.has(number)" class="selection-order">
                {{ selectionStep(number) }}
              </small>
              <small
                v-if="numberState(number) !== 'neutral'"
                class="possible-state-badge"
                aria-hidden="true"
              >{{ numberState(number) === "candidate" ? "C" : numberState(number) === "fixed" ? "🔒" : "×" }}</small>
            </button>
          </div>
          <p class="possible-help">
            Click / Enter: Neutral → Candidate → Fixed · C: Candidate · F: Fixed (maximum six) · X or Alt+click: Excluded · Delete: clear
          </p>
          <div class="possible-draw-state-legend" aria-label="Possible Draw number states">
            <span class="candidate"><b>C</b> Candidate</span>
            <span class="fixed"><b aria-hidden="true">&#128274;</b> Fixed</span>
            <span class="excluded"><b>&times;</b> Excluded</span>
          </div>
          <p v-if="errorMessage" class="possible-draw-action-message" role="status">
            {{ errorMessage }}
          </p>
        </section>

        <section class="possible-analysis-panel">
          <div class="prediction-meters-panel">
            <article
              v-for="strategy in orderedStrategies"
              :key="strategy.id"
              class="prediction-meter-card"
              :style="{ '--meter-color': strategyColor(strategy.id) }"
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
