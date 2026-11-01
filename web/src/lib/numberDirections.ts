import { computed, ref, type Ref } from "vue";
import type { FirstNumberDirectionAnalysis } from "../types";

export const positionLabels = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"];

export function useDirectionSelection(report: Readonly<Ref<FirstNumberDirectionAnalysis | null | undefined>>) {
  const selectedPosition = ref(1);
  const data = computed(() => report.value?.positions.find(item => item.position === selectedPosition.value));
  return { selectedPosition, data };
}
