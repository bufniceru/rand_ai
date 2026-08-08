import { computed, ref, type ComputedRef } from "vue";
import type { ColorTemplate, ColorTokenId, HexColor } from "../types";

export const COLOR_TEMPLATE_KIND = "rand-ai-color-template" as const;
export const COLOR_TEMPLATE_SCHEMA_VERSION = 1 as const;
export const DEFAULT_COLOR_TEMPLATE_NAME = "Rand AI Default";
export const COLOR_TEMPLATE_EXTENSION = ".randai-theme.json";

export type ColorTokenGroupId =
  | "application"
  | "surfaces"
  | "text"
  | "controls"
  | "tables"
  | "numbers"
  | "status"
  | "predictions"
  | "portfolio"
  | "charts"
  | "strategies"
  | "advanced";

export interface ColorTokenDefinition {
  id: ColorTokenId;
  label: string;
  group: ColorTokenGroupId;
  defaultValue: HexColor;
  description?: string;
  cssVariable?: string;
  advanced?: boolean;
}

export interface ColorTemplateValidationResult {
  template: ColorTemplate;
  warnings: string[];
}

export const COLOR_TOKEN_GROUPS: ReadonlyArray<{
  id: ColorTokenGroupId;
  label: string;
  description: string;
}> = [
  { id: "application", label: "Application", description: "Shell, toolbar, tabs, navigation, and status bar." },
  { id: "surfaces", label: "Surfaces", description: "Panels, cards, dialogs, overlays, borders, and shadows." },
  { id: "text", label: "Text", description: "Primary, muted, inverse, links, and disabled text." },
  { id: "controls", label: "Controls", description: "Buttons, inputs, selection, focus, hover, and disabled states." },
  { id: "tables", label: "Tables", description: "Headers, rows, highlights, and separators." },
  { id: "numbers", label: "Numbers", description: "Number cells and actual, predicted, matched, or missed states." },
  { id: "status", label: "Status", description: "Success, warning, error, information, and experimental states." },
  { id: "predictions", label: "Predictions", description: "Prediction, audit, effectiveness, gap, and space displays." },
  { id: "portfolio", label: "Draw Portfolio", description: "Portfolio controls, results, hit buckets, and audit states." },
  { id: "charts", label: "Charts", description: "Plot areas, axes, series, heatmaps, scales, and hover labels." },
  { id: "strategies", label: "Strategies", description: "Strategy families and individual strategy colors." },
  { id: "advanced", label: "Advanced component colors", description: "Fine-grained colors discovered from component styles." },
];

const staticDefinitions: ColorTokenDefinition[] = [
  { id: "application.background", label: "Application background", group: "application", defaultValue: "#E9EFF4" },
  { id: "application.backgroundAccent", label: "Background accent", group: "application", defaultValue: "#2E63C52E" },
  { id: "application.toolbarStart", label: "Toolbar gradient start", group: "application", defaultValue: "#0E2438" },
  { id: "application.toolbarMiddle", label: "Toolbar gradient middle", group: "application", defaultValue: "#173D64" },
  { id: "application.toolbarEnd", label: "Toolbar gradient end", group: "application", defaultValue: "#1F507F" },
  { id: "application.toolbarText", label: "Toolbar text", group: "application", defaultValue: "#FFFFFF" },
  { id: "application.brandStart", label: "Brand gradient start", group: "application", defaultValue: "#FFF1A8" },
  { id: "application.brandEnd", label: "Brand gradient end", group: "application", defaultValue: "#F0B44F" },
  { id: "application.brandText", label: "Brand mark text", group: "application", defaultValue: "#10283F" },
  { id: "application.navigation", label: "Navigation primary", group: "application", defaultValue: "#10283F" },
  { id: "application.navigationRaised", label: "Navigation raised", group: "application", defaultValue: "#173D64" },
  { id: "application.accent", label: "Application accent", group: "application", defaultValue: "#2E63C5" },
  { id: "surfaces.primary", label: "Primary surface", group: "surfaces", defaultValue: "#FFFFFF" },
  { id: "surfaces.muted", label: "Muted surface", group: "surfaces", defaultValue: "#F5F8FB" },
  { id: "surfaces.raised", label: "Raised surface", group: "surfaces", defaultValue: "#F8FAFC" },
  { id: "surfaces.border", label: "Surface border", group: "surfaces", defaultValue: "#D9E2EA" },
  { id: "surfaces.overlay", label: "Modal overlay", group: "surfaces", defaultValue: "#0000007A" },
  { id: "surfaces.shadow", label: "Surface shadow", group: "surfaces", defaultValue: "#06142238" },
  { id: "text.primary", label: "Primary text", group: "text", defaultValue: "#172033" },
  { id: "text.muted", label: "Muted text", group: "text", defaultValue: "#68778D" },
  { id: "text.inverse", label: "Inverse text", group: "text", defaultValue: "#FFFFFF" },
  { id: "text.link", label: "Link text", group: "text", defaultValue: "#2E63C5" },
  { id: "text.disabled", label: "Disabled text", group: "text", defaultValue: "#718197" },
  { id: "controls.primary", label: "Primary control", group: "controls", defaultValue: "#2E63C5" },
  { id: "controls.primaryHover", label: "Primary control hover", group: "controls", defaultValue: "#174B82" },
  { id: "controls.secondary", label: "Secondary control", group: "controls", defaultValue: "#FFFFFF" },
  { id: "controls.secondaryHover", label: "Secondary control hover", group: "controls", defaultValue: "#EEF5FB" },
  { id: "controls.border", label: "Control border", group: "controls", defaultValue: "#C8D8E7" },
  { id: "controls.focus", label: "Focus ring", group: "controls", defaultValue: "#2E63C552" },
  { id: "controls.disabled", label: "Disabled control", group: "controls", defaultValue: "#E8EEF4" },
  { id: "tables.header", label: "Table header", group: "tables", defaultValue: "#173D64" },
  { id: "tables.headerText", label: "Table header text", group: "tables", defaultValue: "#FFFFFF" },
  { id: "tables.row", label: "Table row", group: "tables", defaultValue: "#FFFFFF" },
  { id: "tables.alternate", label: "Alternate table row", group: "tables", defaultValue: "#F5F8FB" },
  { id: "tables.hover", label: "Table row hover", group: "tables", defaultValue: "#EAF4FC" },
  { id: "tables.selected", label: "Selected table row", group: "tables", defaultValue: "#D9EBFB" },
  { id: "numbers.normal", label: "Normal number", group: "numbers", defaultValue: "#FFFFFF" },
  { id: "numbers.predicted", label: "Predicted number", group: "numbers", defaultValue: "#DCEEFF" },
  { id: "numbers.actual", label: "Actual number", group: "numbers", defaultValue: "#FFF3C8" },
  { id: "numbers.matched", label: "Matched number", group: "numbers", defaultValue: "#DFF4E7" },
  { id: "numbers.missed", label: "Missed number", group: "numbers", defaultValue: "#FDE7E9" },
  { id: "numbers.selected", label: "Selected number", group: "numbers", defaultValue: "#2E63C5" },
  { id: "numbers.unavailable", label: "Unavailable number", group: "numbers", defaultValue: "#E8ECEF" },
  { id: "status.success", label: "Success", group: "status", defaultValue: "#2F9E69" },
  { id: "status.warning", label: "Warning", group: "status", defaultValue: "#E49A33" },
  { id: "status.error", label: "Error", group: "status", defaultValue: "#D93A3A" },
  { id: "status.info", label: "Information", group: "status", defaultValue: "#3377B2" },
  { id: "status.experimental", label: "Experimental", group: "status", defaultValue: "#AB9DF2" },
  { id: "predictions.auditNone", label: "Prediction audit: no strategies", group: "predictions", defaultValue: "#49464A" },
  { id: "predictions.auditLow", label: "Prediction audit: 1–2 strategies", group: "predictions", defaultValue: "#78DCE8" },
  { id: "predictions.auditMedium", label: "Prediction audit: 3–5 strategies", group: "predictions", defaultValue: "#A9DC76" },
  { id: "predictions.auditHigh", label: "Prediction audit: 6–8 strategies", group: "predictions", defaultValue: "#AB9DF2" },
  { id: "predictions.auditExtreme", label: "Prediction audit: 9+ strategies", group: "predictions", defaultValue: "#FF6188" },
  { id: "predictions.autocorrelationStrongNegative", label: "Autocorrelation strong negative", group: "predictions", defaultValue: "#78DCE8" },
  { id: "predictions.autocorrelationMildNegative", label: "Autocorrelation mild negative", group: "predictions", defaultValue: "#AB9DF2" },
  { id: "predictions.autocorrelationNeutral", label: "Autocorrelation neutral", group: "predictions", defaultValue: "#A9DC76" },
  { id: "predictions.autocorrelationMildPositive", label: "Autocorrelation mild positive", group: "predictions", defaultValue: "#FFD866" },
  { id: "predictions.autocorrelationStrongPositive", label: "Autocorrelation strong positive", group: "predictions", defaultValue: "#FF6188" },
  { id: "predictions.coOccurrenceLow", label: "Co-occurrence low lift", group: "predictions", defaultValue: "#78DCE8" },
  { id: "predictions.coOccurrenceNormal", label: "Co-occurrence near baseline", group: "predictions", defaultValue: "#A9DC76" },
  { id: "predictions.coOccurrenceElevated", label: "Co-occurrence elevated", group: "predictions", defaultValue: "#FFD866" },
  { id: "predictions.coOccurrenceHigh", label: "Co-occurrence high lift", group: "predictions", defaultValue: "#FF6188" },
  { id: "charts.paper", label: "Chart paper", group: "charts", defaultValue: "#403E41" },
  { id: "charts.plot", label: "Chart plot area", group: "charts", defaultValue: "#403E41" },
  { id: "charts.title", label: "Chart title", group: "charts", defaultValue: "#FCFCFA" },
  { id: "charts.text", label: "Chart text", group: "charts", defaultValue: "#B7B5B7" },
  { id: "charts.grid", label: "Chart grid", group: "charts", defaultValue: "#FCFCFA1F" },
  { id: "charts.zeroLine", label: "Chart zero line", group: "charts", defaultValue: "#5B595C" },
  { id: "charts.hoverBackground", label: "Chart hover background", group: "charts", defaultValue: "#221F22" },
  { id: "charts.hoverBorder", label: "Chart hover border", group: "charts", defaultValue: "#FFD866" },
  { id: "charts.series1", label: "Chart series 1", group: "charts", defaultValue: "#78DCE8" },
  { id: "charts.series2", label: "Chart series 2", group: "charts", defaultValue: "#FF6188" },
  { id: "charts.series3", label: "Chart series 3", group: "charts", defaultValue: "#A9DC76" },
  { id: "charts.series4", label: "Chart series 4", group: "charts", defaultValue: "#FFD866" },
  { id: "charts.series5", label: "Chart series 5", group: "charts", defaultValue: "#AB9DF2" },
  { id: "charts.series6", label: "Chart series 6", group: "charts", defaultValue: "#FC9867" },
  { id: "charts.heatLow", label: "Heatmap low", group: "charts", defaultValue: "#2D2A2E" },
  { id: "charts.heatMidLow", label: "Heatmap lower middle", group: "charts", defaultValue: "#AB9DF2" },
  { id: "charts.heatMid", label: "Heatmap middle", group: "charts", defaultValue: "#78DCE8" },
  { id: "charts.heatMidHigh", label: "Heatmap upper middle", group: "charts", defaultValue: "#A9DC76" },
  { id: "charts.heatHigh", label: "Heatmap high", group: "charts", defaultValue: "#FFD866" },
  { id: "charts.negative", label: "Diverging negative", group: "charts", defaultValue: "#FF6188" },
  { id: "charts.neutral", label: "Diverging neutral", group: "charts", defaultValue: "#403E41" },
  { id: "charts.positive", label: "Diverging positive", group: "charts", defaultValue: "#78DCE8" },
];

const definitionsById = new Map<ColorTokenId, ColorTokenDefinition>();
let definitionRevision = 0;
const definitionRevisionRef = ref(0);

function cssVariableForToken(id: ColorTokenId): string {
  return `--theme-${id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
}

export function registerColorTokens(
  definitions: readonly ColorTokenDefinition[],
): void {
  for (const definition of definitions) {
    const normalized = {
      ...definition,
      defaultValue: normalizeHexColor(definition.defaultValue),
      cssVariable: definition.cssVariable ?? cssVariableForToken(definition.id),
    };
    const previous = definitionsById.get(definition.id);
    if (
      previous &&
      (previous.defaultValue !== normalized.defaultValue ||
        previous.group !== normalized.group)
    ) {
      throw new Error(`Conflicting color token registration: ${definition.id}`);
    }
    if (!previous) {
      definitionsById.set(definition.id, normalized);
      definitionRevision += 1;
    }
  }
  definitionRevisionRef.value = definitionRevision;
}

registerColorTokens(staticDefinitions);

export function colorTokenDefinitions(): ColorTokenDefinition[] {
  return [...definitionsById.values()];
}

export function colorTokenDefinitionRevision(): ComputedRef<number> {
  return computed(() => definitionRevisionRef.value);
}

export function isHexColor(value: unknown): value is HexColor {
  return typeof value === "string" && /^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$/.test(value);
}

export function normalizeHexColor(value: string): HexColor {
  const trimmed = value.trim();
  if (/^#[0-9A-Fa-f]{3}$/.test(trimmed)) {
    return `#${[...trimmed.slice(1)].map((character) => character.repeat(2)).join("")}`.toUpperCase() as HexColor;
  }
  if (/^#[0-9A-Fa-f]{4}$/.test(trimmed)) {
    return `#${[...trimmed.slice(1)].map((character) => character.repeat(2)).join("")}`.toUpperCase() as HexColor;
  }
  if (!isHexColor(trimmed)) throw new Error(`Invalid color value: ${value}`);
  return trimmed.toUpperCase() as HexColor;
}

export function defaultColorMap(): Record<ColorTokenId, HexColor> {
  return Object.fromEntries(
    colorTokenDefinitions().map((definition) => [
      definition.id,
      definition.defaultValue,
    ]),
  );
}

export function createDefaultColorTemplate(): ColorTemplate {
  return {
    kind: COLOR_TEMPLATE_KIND,
    schemaVersion: COLOR_TEMPLATE_SCHEMA_VERSION,
    name: DEFAULT_COLOR_TEMPLATE_NAME,
    description: "The built-in Rand AI color template.",
    createdWith: "Rand AI 0.1.0",
    colors: defaultColorMap(),
  };
}

export function cloneColorTemplate(template: ColorTemplate): ColorTemplate {
  return { ...template, colors: { ...template.colors } };
}

export function materializeColorTemplate(template: ColorTemplate): ColorTemplate {
  return {
    ...cloneColorTemplate(template),
    colors: { ...defaultColorMap(), ...template.colors },
  };
}

export function validateColorTemplate(
  input: unknown,
): ColorTemplateValidationResult {
  if (!input || typeof input !== "object") {
    throw new Error("The selected file does not contain a color template.");
  }
  const value = input as Record<string, unknown>;
  if (value.kind !== COLOR_TEMPLATE_KIND) {
    throw new Error(`Template kind must be ${COLOR_TEMPLATE_KIND}.`);
  }
  if (value.schemaVersion !== COLOR_TEMPLATE_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported color-template schema version: ${String(value.schemaVersion)}.`,
    );
  }
  if (typeof value.name !== "string" || !value.name.trim() || value.name.length > 80) {
    throw new Error("Template name must contain 1 to 80 characters.");
  }
  if (value.description !== undefined && (typeof value.description !== "string" || value.description.length > 500)) {
    throw new Error("Template description must contain at most 500 characters.");
  }
  if (!value.colors || typeof value.colors !== "object" || Array.isArray(value.colors)) {
    throw new Error("Template colors must be an object.");
  }

  const known = new Set(colorTokenDefinitions().map((definition) => definition.id));
  const supplied: Record<ColorTokenId, HexColor> = {};
  const unknown: string[] = [];
  for (const [id, color] of Object.entries(value.colors)) {
    if (!/^[a-zA-Z0-9_.-]{1,160}$/.test(id)) {
      throw new Error(`Invalid color token identifier: ${id}`);
    }
    if (!isHexColor(color)) {
      throw new Error(`Color token ${id} must use #RRGGBB or #RRGGBBAA.`);
    }
    if (known.has(id)) supplied[id] = normalizeHexColor(color);
    else unknown.push(id);
  }

  const missing = [...known].filter((id) => supplied[id] === undefined);
  const warnings: string[] = [];
  if (missing.length) warnings.push(`${missing.length} missing color token${missing.length === 1 ? " was" : "s were"} restored from Rand AI Default.`);
  if (unknown.length) warnings.push(`${unknown.length} unknown color token${unknown.length === 1 ? " was" : "s were"} ignored.`);

  const optionalString = (key: "createdWith" | "exportedAt"): string | undefined =>
    typeof value[key] === "string" ? value[key] as string : undefined;
  return {
    template: {
      kind: COLOR_TEMPLATE_KIND,
      schemaVersion: COLOR_TEMPLATE_SCHEMA_VERSION,
      name: value.name.trim(),
      description: typeof value.description === "string" ? value.description.trim() : undefined,
      createdWith: optionalString("createdWith"),
      exportedAt: optionalString("exportedAt"),
      colors: { ...defaultColorMap(), ...supplied },
    },
    warnings,
  };
}

const activeTemplateRef = ref<ColorTemplate>(createDefaultColorTemplate());
const themeRevisionRef = ref(0);
let previewSnapshot: ColorTemplate | null = null;

export const activeColorTemplate = computed(() => activeTemplateRef.value);
export const themeRevision = computed(() => themeRevisionRef.value);

export function themeColor(id: ColorTokenId, fallback = "#727072"): HexColor {
  return activeTemplateRef.value.colors[id]
    ?? definitionsById.get(id)?.defaultValue
    ?? normalizeHexColor(fallback);
}

function applyTemplateToDocument(template: ColorTemplate): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  for (const definition of colorTokenDefinitions()) {
    root.style.setProperty(
      definition.cssVariable ?? cssVariableForToken(definition.id),
      template.colors[definition.id] ?? definition.defaultValue,
    );
  }
  root.dataset.colorTemplate = template.name;
  root.style.colorScheme = "light dark";
}

export function setActiveColorTemplate(template: ColorTemplate): void {
  const materialized = materializeColorTemplate(template);
  activeTemplateRef.value = materialized;
  applyTemplateToDocument(materialized);
  themeRevisionRef.value += 1;
}

export function beginColorTemplatePreview(): ColorTemplate {
  previewSnapshot = cloneColorTemplate(activeTemplateRef.value);
  return cloneColorTemplate(activeTemplateRef.value);
}

export function previewColorTemplate(template: ColorTemplate): void {
  setActiveColorTemplate(template);
}

export function cancelColorTemplatePreview(): void {
  if (previewSnapshot) setActiveColorTemplate(previewSnapshot);
  previewSnapshot = null;
}

export function commitColorTemplatePreview(template: ColorTemplate): void {
  previewSnapshot = null;
  setActiveColorTemplate(template);
}

function hashTokenSeed(seed: string): string {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function rgbaToHex(value: string): HexColor | null {
  const match = value.match(/^rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(?:\s*,\s*(\d*\.?\d+)\s*)?\)$/i);
  if (!match) return null;
  const channels = match.slice(1, 4).map((channel) => Math.max(0, Math.min(255, Math.round(Number(channel)))));
  const alpha = match[4] === undefined ? 255 : Math.max(0, Math.min(255, Math.round(Number(match[4]) * 255)));
  return `#${[...channels, ...(alpha < 255 ? [alpha] : [])].map((channel) => channel.toString(16).padStart(2, "0")).join("")}`.toUpperCase() as HexColor;
}

const sourceColorPattern = /#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)/g;

function inferAdvancedGroup(selector: string): ColorTokenGroupId {
  const normalized = selector.toLowerCase();
  if (normalized.includes("portfolio")) return "portfolio";
  if (normalized.includes("prediction") || normalized.includes("audit") || normalized.includes("last-seen")) return "predictions";
  if (normalized.includes("number")) return "numbers";
  if (normalized.includes("table") || normalized.includes("row") || normalized.includes("cell")) return "tables";
  if (normalized.includes("button") || normalized.includes("input") || normalized.includes("select") || normalized.includes("control")) return "controls";
  if (normalized.includes("status") || normalized.includes("warning") || normalized.includes("error") || normalized.includes("success")) return "status";
  if (normalized.includes("dialog") || normalized.includes("card") || normalized.includes("panel") || normalized.includes("surface")) return "surfaces";
  if (normalized.includes("toolbar") || normalized.includes("tabs") || normalized.includes("app-")) return "application";
  return "advanced";
}

function humanSelector(selector: string): string {
  return selector.replace(/\[[^\]]+\]/g, "").replace(/\s+/g, " ").trim().slice(0, 86);
}

function discoverRuleColors(rule: CSSRule, path: string): void {
  if (typeof CSSStyleRule !== "undefined" && rule instanceof CSSStyleRule) {
    const selector = rule.selectorText;
    for (const property of [...rule.style]) {
      const value = rule.style.getPropertyValue(property);
      if (!value || value.includes("var(--theme-")) continue;
      let occurrence = 0;
      const replaced = value.replace(sourceColorPattern, (source) => {
        const normalized = source.startsWith("#") ? normalizeHexColor(source) : rgbaToHex(source);
        if (!normalized) return source;
        const seed = `${path}|${selector}|${property}|${occurrence}`;
        occurrence += 1;
        const id = `advanced.css.${hashTokenSeed(seed)}`;
        registerColorTokens([{
          id,
          label: `${humanSelector(selector)} · ${property}${occurrence > 1 ? ` ${occurrence}` : ""}`,
          group: inferAdvancedGroup(selector),
          defaultValue: normalized,
          description: `Advanced override for ${selector} (${property}).`,
          advanced: true,
        }]);
        return `var(${cssVariableForToken(id)})`;
      });
      if (replaced !== value) {
        rule.style.setProperty(property, replaced, rule.style.getPropertyPriority(property));
      }
    }
  }
  const grouping = rule as CSSGroupingRule;
  if (grouping.cssRules) {
    [...grouping.cssRules].forEach((child, index) => discoverRuleColors(child, `${path}.${index}`));
  }
}

export function initializeColorTemplateRuntime(): void {
  if (typeof document === "undefined") return;
  [...document.styleSheets].forEach((sheet, sheetIndex) => {
    try {
      [...sheet.cssRules].forEach((rule, ruleIndex) => discoverRuleColors(rule, `${sheetIndex}.${ruleIndex}`));
    } catch {
      // Cross-origin stylesheets are intentionally left untouched.
    }
  });
  setActiveColorTemplate(activeTemplateRef.value);
}

export function contrastRatio(foreground: HexColor, background: HexColor): number {
  const channels = (color: HexColor): number[] => [1, 3, 5].map((offset) => Number.parseInt(color.slice(offset, offset + 2), 16) / 255);
  const luminance = (color: HexColor): number => {
    const [red, green, blue] = channels(color).map((channel) => channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  };
  const high = Math.max(luminance(foreground), luminance(background));
  const low = Math.min(luminance(foreground), luminance(background));
  return (high + 0.05) / (low + 0.05);
}
