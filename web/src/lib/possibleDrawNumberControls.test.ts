import { computed, effectScope, ref } from "vue";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { usePossibleDrawNumberControls } from "./possibleDrawNumberControls";
import { configurePossibleDrawContext, getPossibleDrawNumberState, resetPossibleDrawStoreForTests } from "./possibleDrawPlans";

let scope = effectScope();
const editable = ref(true);
const showError = vi.fn();
beforeEach(() => {
  scope = effectScope();
  editable.value = true;
  showError.mockReset();
  resetPossibleDrawStoreForTests();
  configurePossibleDrawContext({ datasetId: "controls", targetDrawId: "3" });
  vi.stubGlobal("window", { randAiDesktop: { showForSureLimitError: showError } });
});
afterEach(() => { scope.stop(); vi.unstubAllGlobals(); });
function controls() {
  return scope.run(() => usePossibleDrawNumberControls(computed(() => editable.value)))!;
}
function click(altKey = false) {
  return { altKey, preventDefault: vi.fn() } as unknown as MouseEvent;
}
function key(value: string) {
  return { key: value, preventDefault: vi.fn(), stopPropagation: vi.fn() } as unknown as KeyboardEvent;
}
it("supports clicking, cycling, Alt exclusion and all selection keys", () => {
  const control = controls();
  for (const state of ["candidate", "fixed", "neutral"]) {
    control.handleHeaderClick(click(), 1);
    expect(control.numberState(1)).toBe(state);
  }
  control.handleHeaderClick(click(true), 1);
  expect(control.numberState(1)).toBe("excluded");
  control.handleHeaderClick(click(), 1);
  expect(control.numberState(1)).toBe("excluded");
  control.handleHeaderClick(click(true), 1);
  expect(control.numberState(1)).toBe("neutral");
  for (const [input, state] of [["Enter", "candidate"], [" ", "fixed"], ["Delete", "neutral"], ["C", "candidate"], ["F", "fixed"], ["X", "excluded"], ["Backspace", "neutral"]]) {
    const event = key(input!);
    control.handleHeaderKeydown(event, 1);
    expect(control.numberState(1)).toBe(state);
    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(event.stopPropagation).toHaveBeenCalledOnce();
  }
  const ignored = key("a");
  control.handleHeaderKeydown(ignored, 1);
  expect(ignored.preventDefault).not.toHaveBeenCalled();
});
it("shares changes and preserves the fixed-number limit and error feedback", () => {
  const first = controls();
  const second = controls();
  for (let number = 1; number <= 6; number++) first.handleHeaderKeydown(key("f"), number);
  expect(second.numberState(1)).toBe("fixed");
  second.handleHeaderKeydown(key("f"), 7);
  expect(second.numberActionMessage.value).toContain("at most six Fixed");
  expect(showError).toHaveBeenCalledWith(7);
  expect(getPossibleDrawNumberState(7)).toBe("neutral");
  configurePossibleDrawContext({ datasetId: "new-context", targetDrawId: "3" });
  expect(second.numberActionMessage.value).toBe("");
  expect(first.numberState(1)).toBe("neutral");
});
it("does not modify selections when editing is disabled", () => {
  const control = controls();
  editable.value = false;
  control.handleHeaderClick(click(), 1);
  control.handleHeaderKeydown(key("f"), 2);
  expect(control.numberState(1)).toBe("neutral");
  expect(control.numberState(2)).toBe("neutral");
  expect(showError).not.toHaveBeenCalled();
});
