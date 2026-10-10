import { ref, watch, type ComputedRef } from "vue";
import {
  activePossibleDrawPlanId,
  cyclePossibleDrawNumberState,
  getPossibleDrawNumberState,
  possibleDrawPlanRevision,
  setPossibleDrawNumberState,
  togglePossibleDrawExcluded,
} from "./possibleDrawPlans";

export const POSSIBLE_DRAW_SELECTION_HELP =
  "Possible Draw: click to cycle Neutral → Candidate → Fixed; Alt-click to exclude. Keys: C candidate, F fixed, X exclude, Delete clear.";
export const NUMBER_BALL_RADIUS = 13.5;

export function usePossibleDrawNumberControls(editable: ComputedRef<boolean>) {
  const numberActionMessage = ref("");
  watch(activePossibleDrawPlanId, () => { numberActionMessage.value = ""; }, { flush: "sync" });

  function numberState(number: number) {
    possibleDrawPlanRevision.value;
    return getPossibleDrawNumberState(number);
  }

  function applyNumberAction(number: number, result: { ok: boolean; message?: string }): void {
    numberActionMessage.value = result.message ?? "";
    if (!result.ok && result.message && typeof window !== "undefined") {
      void window.randAiDesktop?.showForSureLimitError(number);
    }
  }

  function handleHeaderClick(event: MouseEvent, number: number): void {
    if (!editable.value) return;
    event.preventDefault();
    applyNumberAction(number, event.altKey
      ? togglePossibleDrawExcluded(number)
      : cyclePossibleDrawNumberState(number));
  }

  function handleHeaderKeydown(event: KeyboardEvent, number: number): void {
    if (!editable.value) return;
    const key = event.key.toLowerCase();
    let result: { ok: boolean; message?: string } | null = null;
    if (key === "enter" || key === " ") result = cyclePossibleDrawNumberState(number);
    if (key === "c") result = setPossibleDrawNumberState(number, "candidate");
    if (key === "f") result = setPossibleDrawNumberState(number, "fixed");
    if (key === "x") result = togglePossibleDrawExcluded(number);
    if (key === "delete" || key === "backspace") result = setPossibleDrawNumberState(number, "neutral");
    if (!result) return;
    event.preventDefault();
    event.stopPropagation();
    applyNumberAction(number, result);
  }

  return { numberActionMessage, numberState, handleHeaderClick, handleHeaderKeydown };
}
