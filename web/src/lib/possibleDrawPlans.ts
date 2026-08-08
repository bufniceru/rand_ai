import { computed, reactive, ref } from "vue";

import type {
  DrawPortfolioMode,
  PossibleDrawNumberState,
  PossibleDrawPlan,
  PossibleDrawPlanContext,
} from "../types";

const PLAN_STORAGE_KEY = "rand-ai.possible-draw.plans.v3";
const LEGACY_PLAN_STORAGE_KEY = "rand-ai.possible-draw.plans.v2";
const MODE_STORAGE_KEY = "rand-ai.draw-portfolio.modes.v1";
const NUMBER_COUNT = 49;
const FIXED_LIMIT = 6;

interface PlanCollection {
  activePlanId: string;
  plans: PossibleDrawPlan[];
}

interface StoredPlanState {
  contexts: Record<string, PlanCollection>;
}

interface LegacyPlan {
  id?: unknown;
  name?: unknown;
  selected?: unknown;
  uncertain?: unknown;
  dropped?: unknown;
}

const state = reactive<StoredPlanState>({ contexts: {} });
const activeContextKey = ref("");
const activeDatasetId = ref("");
const revision = ref(0);
const modePreferences = reactive<Record<string, DrawPortfolioMode>>({});
let storageLoaded = false;

function availableStorage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

function validNumbers(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value)]
    .filter(
      (number): number is number =>
        typeof number === "number" &&
        Number.isInteger(number) &&
        number >= 1 &&
        number <= NUMBER_COUNT,
    );
}

function normalizePlan(value: Partial<PossibleDrawPlan>, fallbackName: string): PossibleDrawPlan {
  const fixedNumbers = validNumbers(value.fixedNumbers).slice(0, FIXED_LIMIT);
  const fixed = new Set(fixedNumbers);
  const candidateNumbers = validNumbers(value.candidateNumbers).filter(
    (number) => !fixed.has(number),
  );
  const candidate = new Set(candidateNumbers);
  const excludedNumbers = validNumbers(value.excludedNumbers).filter(
    (number) => !fixed.has(number) && !candidate.has(number),
  );
  return {
    id: typeof value.id === "string" && value.id ? value.id : createId(),
    name: typeof value.name === "string" && value.name.trim()
      ? value.name.trim().slice(0, 80)
      : fallbackName,
    fixedNumbers,
    candidateNumbers,
    excludedNumbers,
  };
}

function createId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newPlan(name = "Draw 1"): PossibleDrawPlan {
  return normalizePlan({ id: createId(), name }, name);
}

function contextKey(context: PossibleDrawPlanContext): string {
  return JSON.stringify([context.datasetId, context.targetDrawId]);
}

function loadStorage(): void {
  if (storageLoaded) return;
  storageLoaded = true;
  const storage = availableStorage();
  if (!storage) return;
  try {
    const raw = JSON.parse(storage.getItem(PLAN_STORAGE_KEY) ?? "null") as
      | { contexts?: unknown }
      | null;
    if (raw?.contexts && typeof raw.contexts === "object") {
      for (const [key, collectionValue] of Object.entries(raw.contexts)) {
        if (!collectionValue || typeof collectionValue !== "object") continue;
        const collection = collectionValue as {
          activePlanId?: unknown;
          plans?: unknown;
        };
        if (!Array.isArray(collection.plans)) continue;
        const plans = collection.plans.map((plan, index) =>
          normalizePlan(
            plan && typeof plan === "object" ? (plan as Partial<PossibleDrawPlan>) : {},
            `Draw ${index + 1}`,
          ),
        );
        if (plans.length === 0) continue;
        state.contexts[key] = {
          activePlanId:
            typeof collection.activePlanId === "string" &&
            plans.some((plan) => plan.id === collection.activePlanId)
              ? collection.activePlanId
              : plans[0].id,
          plans,
        };
      }
    }
  } catch {
    // A malformed saved plan must not prevent the application from opening.
  }
  try {
    const rawModes = JSON.parse(storage.getItem(MODE_STORAGE_KEY) ?? "null") as
      | Record<string, unknown>
      | null;
    if (rawModes && typeof rawModes === "object") {
      for (const [datasetId, mode] of Object.entries(rawModes)) {
        if (mode === "classic" || mode === "guided") modePreferences[datasetId] = mode;
      }
    }
  } catch {
    // Ignore malformed preferences and use Classic.
  }
}

function migrateLegacyCollection(): PlanCollection | null {
  const storage = availableStorage();
  if (!storage) return null;
  try {
    const raw = JSON.parse(storage.getItem(LEGACY_PLAN_STORAGE_KEY) ?? "null") as
      | { activePlanId?: unknown; plans?: unknown }
      | null;
    if (!raw || !Array.isArray(raw.plans) || raw.plans.length === 0) return null;
    const plans = (raw.plans as LegacyPlan[]).map((legacy, index) =>
      normalizePlan(
        {
          id: typeof legacy.id === "string" ? legacy.id : undefined,
          name: typeof legacy.name === "string" ? legacy.name : undefined,
          fixedNumbers: validNumbers(legacy.selected),
          candidateNumbers: validNumbers(legacy.uncertain),
          excludedNumbers: validNumbers(legacy.dropped),
        },
        `Draw ${index + 1}`,
      ),
    );
    return {
      activePlanId:
        typeof raw.activePlanId === "string" &&
        plans.some((plan) => plan.id === raw.activePlanId)
          ? raw.activePlanId
          : plans[0].id,
      plans,
    };
  } catch {
    return null;
  }
}

function savePlans(): void {
  availableStorage()?.setItem(PLAN_STORAGE_KEY, JSON.stringify(state));
  revision.value += 1;
}

function saveModes(): void {
  availableStorage()?.setItem(MODE_STORAGE_KEY, JSON.stringify(modePreferences));
}

function currentCollection(): PlanCollection | null {
  return state.contexts[activeContextKey.value] ?? null;
}

function currentPlan(): PossibleDrawPlan | null {
  const collection = currentCollection();
  return collection?.plans.find((plan) => plan.id === collection.activePlanId) ?? null;
}

function replaceCurrentPlan(plan: PossibleDrawPlan): void {
  const collection = currentCollection();
  if (!collection) return;
  const index = collection.plans.findIndex((entry) => entry.id === collection.activePlanId);
  if (index < 0) return;
  collection.plans[index] = normalizePlan(plan, collection.plans[index].name);
  savePlans();
}

export const possibleDrawPlans = computed(() => currentCollection()?.plans ?? []);
export const activePossibleDrawPlanId = computed(() => currentCollection()?.activePlanId ?? "");
export const activePossibleDrawPlan = computed(() => currentPlan());
export const possibleDrawPlanRevision = computed(() => revision.value);
export const activePossibleDrawState = computed(() => ({
  fixedNumbers: [...(currentPlan()?.fixedNumbers ?? [])],
  candidateNumbers: [...(currentPlan()?.candidateNumbers ?? [])],
  excludedNumbers: [...(currentPlan()?.excludedNumbers ?? [])],
}));

export function configurePossibleDrawContext(context: PossibleDrawPlanContext): void {
  loadStorage();
  activeDatasetId.value = context.datasetId;
  const key = contextKey(context);
  if (!state.contexts[key]) {
    state.contexts[key] =
      Object.keys(state.contexts).length === 0
        ? migrateLegacyCollection() ?? { activePlanId: "", plans: [] }
        : { activePlanId: "", plans: [] };
    if (state.contexts[key].plans.length === 0) {
      const plan = newPlan();
      state.contexts[key].plans = [plan];
      state.contexts[key].activePlanId = plan.id;
    }
    savePlans();
  }
  activeContextKey.value = key;
  revision.value += 1;
}

export function getPossibleDrawNumberState(number: number): PossibleDrawNumberState {
  const plan = currentPlan();
  if (!plan) return "neutral";
  if (plan.fixedNumbers.includes(number)) return "fixed";
  if (plan.candidateNumbers.includes(number)) return "candidate";
  if (plan.excludedNumbers.includes(number)) return "excluded";
  return "neutral";
}

export function setPossibleDrawNumberState(
  number: number,
  nextState: PossibleDrawNumberState,
): { ok: boolean; message?: string } {
  const plan = currentPlan();
  if (!plan || !Number.isInteger(number) || number < 1 || number > NUMBER_COUNT) {
    return { ok: false, message: "The number is outside the Possible Draw range." };
  }
  if (
    nextState === "fixed" &&
    !plan.fixedNumbers.includes(number) &&
    plan.fixedNumbers.length >= FIXED_LIMIT
  ) {
    return { ok: false, message: "A Possible Draw can contain at most six Fixed numbers." };
  }
  const updated: PossibleDrawPlan = {
    ...plan,
    fixedNumbers: plan.fixedNumbers.filter((entry) => entry !== number),
    candidateNumbers: plan.candidateNumbers.filter((entry) => entry !== number),
    excludedNumbers: plan.excludedNumbers.filter((entry) => entry !== number),
  };
  if (nextState === "fixed") updated.fixedNumbers.push(number);
  if (nextState === "candidate") updated.candidateNumbers.push(number);
  if (nextState === "excluded") updated.excludedNumbers.push(number);
  replaceCurrentPlan(updated);
  return { ok: true };
}

export function cyclePossibleDrawNumberState(
  number: number,
): { ok: boolean; message?: string } {
  const current = getPossibleDrawNumberState(number);
  if (current === "excluded") return { ok: true };
  const next: PossibleDrawNumberState =
    current === "neutral" ? "candidate" : current === "candidate" ? "fixed" : "neutral";
  return setPossibleDrawNumberState(number, next);
}

export function togglePossibleDrawExcluded(number: number): { ok: boolean; message?: string } {
  return setPossibleDrawNumberState(
    number,
    getPossibleDrawNumberState(number) === "excluded" ? "neutral" : "excluded",
  );
}

export function replacePossibleDrawNumbers(
  field: "fixedNumbers" | "candidateNumbers" | "excludedNumbers",
  numbers: number[],
): void {
  const plan = currentPlan();
  if (!plan) return;
  replaceCurrentPlan({ ...plan, [field]: numbers });
}

export function createPossibleDrawPlan(): void {
  const collection = currentCollection();
  if (!collection) return;
  const plan = newPlan(`Draw ${collection.plans.length + 1}`);
  collection.plans.push(plan);
  collection.activePlanId = plan.id;
  savePlans();
}

export function selectPossibleDrawPlan(planId: string): void {
  const collection = currentCollection();
  if (!collection?.plans.some((plan) => plan.id === planId)) return;
  collection.activePlanId = planId;
  savePlans();
}

export function deletePossibleDrawPlan(): void {
  const collection = currentCollection();
  if (!collection) return;
  if (collection.plans.length <= 1) {
    replaceCurrentPlan({ ...collection.plans[0], fixedNumbers: [], candidateNumbers: [], excludedNumbers: [] });
    return;
  }
  const index = collection.plans.findIndex((plan) => plan.id === collection.activePlanId);
  collection.plans.splice(Math.max(index, 0), 1);
  collection.activePlanId = collection.plans[Math.min(Math.max(index, 0), collection.plans.length - 1)].id;
  savePlans();
}

export function resetPossibleDrawPlan(): void {
  const plan = currentPlan();
  if (!plan) return;
  replaceCurrentPlan({ ...plan, fixedNumbers: [], candidateNumbers: [], excludedNumbers: [] });
}

export function getDrawPortfolioMode(datasetId = activeDatasetId.value): DrawPortfolioMode {
  loadStorage();
  return modePreferences[datasetId] ?? "classic";
}

export function setDrawPortfolioMode(datasetId: string, mode: DrawPortfolioMode): void {
  loadStorage();
  modePreferences[datasetId] = mode;
  saveModes();
}

export function resetPossibleDrawStoreForTests(): void {
  Object.keys(state.contexts).forEach((key) => delete state.contexts[key]);
  Object.keys(modePreferences).forEach((key) => delete modePreferences[key]);
  activeContextKey.value = "";
  activeDatasetId.value = "";
  revision.value = 0;
  storageLoaded = false;
}
