<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import {
  COLOR_TOKEN_GROUPS,
  cloneColorTemplate,
  colorTokenDefinitionRevision,
  colorTokenDefinitions,
  contrastRatio,
  createDefaultColorTemplate,
  isHexColor,
  materializeColorTemplate,
  normalizeHexColor,
  validateColorTemplate,
  type ColorTokenDefinition,
  type ColorTokenGroupId,
} from "../lib/colorTemplates";
import type { ColorTemplate, HexColor } from "../types";

const props = defineProps<{
  activeTemplate: ColorTemplate;
  saving: boolean;
  externalError?: string;
}>();

const emit = defineEmits<{
  cancel: [];
  preview: [template: ColorTemplate];
  apply: [template: ColorTemplate];
}>();

const dialog = ref<HTMLElement | null>(null);
const draft = ref<ColorTemplate>(cloneColorTemplate(props.activeTemplate));
const original = cloneColorTemplate(props.activeTemplate);
const query = ref("");
const notice = ref("");
const errorMessage = ref("");
const fileBusy = ref(false);
const openGroups = ref<Set<ColorTokenGroupId>>(new Set(["application"]));
const definitionRevision = colorTokenDefinitionRevision();
const textValues = ref<Record<string, string>>({ ...draft.value.colors });
const defaultTemplate = createDefaultColorTemplate();

const definitions = computed(() => {
  definitionRevision.value;
  return colorTokenDefinitions();
});

const visibleGroups = computed(() => {
  const normalizedQuery = query.value.trim().toLowerCase();
  return COLOR_TOKEN_GROUPS.map((group) => ({
    ...group,
    tokens: definitions.value.filter((token) => {
      if (token.group !== group.id) return false;
      if (!normalizedQuery) return true;
      return [token.id, token.label, token.description, group.label]
        .filter(Boolean)
        .some((value) => value!.toLowerCase().includes(normalizedQuery));
    }),
  })).filter((group) => group.tokens.length > 0);
});

const contrastWarnings = computed(() => {
  const pairs: Array<[string, string, string]> = [
    ["text.primary", "surfaces.primary", "Primary text on the main surface"],
    ["text.muted", "surfaces.primary", "Muted text on the main surface"],
    ["application.toolbarText", "application.toolbarMiddle", "Toolbar text"],
    ["text.inverse", "controls.primary", "Primary button text"],
    ["tables.headerText", "tables.header", "Table header text"],
  ];
  return pairs.flatMap(([foreground, background, label]) => {
    const ratio = contrastRatio(
      colorFor(foreground),
      colorFor(background),
    );
    return ratio < 4.5 ? [`${label}: ${ratio.toFixed(2)}:1`] : [];
  });
});

watch(
  draft,
  (value) => emit("preview", materializeColorTemplate(value)),
  { deep: true },
);

watch(query, (value) => {
  if (value.trim()) {
    openGroups.value = new Set(visibleGroups.value.map((group) => group.id));
  }
});

function colorFor(id: string): HexColor {
  return draft.value.colors[id]
    ?? definitions.value.find((definition) => definition.id === id)?.defaultValue
    ?? "#727072";
}

function rgbPart(color: HexColor): string {
  return color.slice(0, 7);
}

function alphaPercent(color: HexColor): number {
  if (color.length === 7) return 100;
  return Math.round((Number.parseInt(color.slice(7, 9), 16) / 255) * 100);
}

function withAlpha(color: HexColor, percent: number): HexColor {
  const alpha = Math.max(0, Math.min(255, Math.round((percent / 100) * 255)));
  return `${color.slice(0, 7)}${alpha === 255 ? "" : alpha.toString(16).padStart(2, "0")}`.toUpperCase() as HexColor;
}

function setColor(id: string, value: string): void {
  try {
    const normalized = normalizeHexColor(value);
    draft.value.colors[id] = normalized;
    textValues.value[id] = normalized;
    errorMessage.value = "";
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
    textValues.value[id] = colorFor(id);
  }
}

function setRgb(id: string, value: string): void {
  const existing = colorFor(id);
  setColor(id, `${value}${existing.length === 9 ? existing.slice(7, 9) : ""}`);
}

function editHex(id: string, value: string): void {
  textValues.value[id] = value;
  if (isHexColor(value)) setColor(id, value);
}

function setAlpha(id: string, value: number): void {
  setColor(id, withAlpha(colorFor(id), value));
}

function resetToken(token: ColorTokenDefinition): void {
  setColor(token.id, token.defaultValue);
}

function resetGroup(groupId: ColorTokenGroupId): void {
  for (const token of definitions.value.filter((item) => item.group === groupId)) {
    draft.value.colors[token.id] = token.defaultValue;
    textValues.value[token.id] = token.defaultValue;
  }
}

function resetAll(): void {
  const defaults = createDefaultColorTemplate();
  draft.value = {
    ...defaults,
    name: draft.value.name,
    description: draft.value.description,
  };
  textValues.value = { ...draft.value.colors };
  notice.value = "All colors were reset to Rand AI Default.";
}

function toggleGroup(groupId: ColorTokenGroupId, open: boolean): void {
  const next = new Set(openGroups.value);
  if (open) next.add(groupId);
  else next.delete(groupId);
  openGroups.value = next;
}

function swatchStyle(color: HexColor): Record<string, string> {
  return { "--template-swatch": color };
}

async function loadTemplate(): Promise<void> {
  if (!window.randAiDesktop) return;
  fileBusy.value = true;
  errorMessage.value = "";
  notice.value = "";
  try {
    const result = await window.randAiDesktop.loadColorTemplate();
    if (result.canceled || result.template === undefined) return;
    const validated = validateColorTemplate(result.template);
    draft.value = validated.template;
    textValues.value = { ...validated.template.colors };
    notice.value = [
      `Loaded ${result.path ?? validated.template.name}.`,
      ...validated.warnings,
    ].join(" ");
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    fileBusy.value = false;
  }
}

async function saveTemplate(): Promise<void> {
  if (!window.randAiDesktop) return;
  fileBusy.value = true;
  errorMessage.value = "";
  notice.value = "";
  try {
    const template = materializeColorTemplate({
      ...draft.value,
      name: draft.value.name.trim() || "Custom Rand AI Template",
      exportedAt: new Date().toISOString(),
    });
    const result = await window.randAiDesktop.saveColorTemplate(template);
    if (!result.canceled) notice.value = `Saved ${result.path ?? template.name}.`;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    fileBusy.value = false;
  }
}

function applyTemplate(): void {
  const name = draft.value.name.trim();
  if (!name || name.length > 80) {
    errorMessage.value = "Template name must contain 1 to 80 characters.";
    return;
  }
  emit("apply", materializeColorTemplate({ ...draft.value, name }));
}

onMounted(() => dialog.value?.focus());
</script>

<template>
  <div class="modal-backdrop appearance-backdrop" role="presentation" @click.self="!saving && !fileBusy && emit('cancel')">
    <section
      ref="dialog"
      class="appearance-dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="appearance-title"
      tabindex="-1"
      @keydown.esc="!saving && !fileBusy && emit('cancel')"
    >
      <header class="appearance-heading">
        <div>
          <p class="eyebrow">Color template designer</p>
          <h2 id="appearance-title">Appearance</h2>
          <p>Customize named roles or open the advanced groups for component-level control.</p>
        </div>
        <div class="appearance-template-actions">
          <button class="button secondary" type="button" :disabled="saving || fileBusy" @click="loadTemplate">Load template</button>
          <button class="button secondary" type="button" :disabled="saving || fileBusy" @click="saveTemplate">Save template</button>
        </div>
      </header>

      <section class="appearance-metadata">
        <label>
          <span>Template name</span>
          <input v-model="draft.name" maxlength="80" :disabled="saving || fileBusy">
        </label>
        <label>
          <span>Description</span>
          <input v-model="draft.description" maxlength="500" placeholder="Optional" :disabled="saving || fileBusy">
        </label>
        <label class="appearance-search">
          <span>Find an element or token</span>
          <input v-model="query" type="search" placeholder="Toolbar, matched, chart grid…">
        </label>
      </section>

      <div class="appearance-body">
        <aside class="appearance-preview" aria-label="Live theme preview">
          <div class="appearance-preview-toolbar">
            <b>Rand AI</b><span>Preview</span>
          </div>
          <div class="appearance-preview-panel">
            <h3>Representative elements</h3>
            <p>Text, controls, number states, and charts update immediately.</p>
            <div class="appearance-preview-buttons">
              <button type="button">Primary</button><button type="button">Secondary</button>
            </div>
            <div class="appearance-preview-numbers">
              <b>7</b><b class="predicted">18</b><b class="matched">32</b><b class="missed">44</b>
            </div>
            <div class="appearance-preview-chart">
              <i v-for="index in 6" :key="index" :style="{ background: colorFor(`charts.series${index}`), height: `${24 + index * 8}px` }" />
            </div>
          </div>
          <div v-if="contrastWarnings.length" class="appearance-contrast-warning">
            <strong>Contrast warnings</strong>
            <span v-for="warning in contrastWarnings" :key="warning">{{ warning }}</span>
          </div>
          <div class="appearance-preview-summary">
            <span>{{ definitions.length.toLocaleString() }} editable colors</span>
            <button type="button" @click="resetAll">Reset all</button>
          </div>
        </aside>

        <div class="appearance-token-groups">
          <details
            v-for="group in visibleGroups"
            :key="group.id"
            :open="openGroups.has(group.id)"
            @toggle="toggleGroup(group.id, ($event.currentTarget as HTMLDetailsElement).open)"
          >
            <summary>
              <span><strong>{{ group.label }}</strong><small>{{ group.description }}</small></span>
              <b>{{ group.tokens.length }}</b>
            </summary>
            <template v-if="openGroups.has(group.id)">
              <div class="appearance-group-actions">
                <button type="button" @click="resetGroup(group.id)">Reset group</button>
              </div>
              <div class="appearance-token-list">
                <article v-for="token in group.tokens" :key="token.id" class="appearance-token">
                <div class="appearance-token-label">
                  <strong>{{ token.label }}</strong>
                  <code>{{ token.id }}</code>
                  <small v-if="token.description">{{ token.description }}</small>
                </div>
                <div class="appearance-token-editor">
                  <span class="template-swatch" :style="swatchStyle(colorFor(token.id))" title="Draft color" />
                  <input
                    type="color"
                    :value="rgbPart(colorFor(token.id))"
                    :aria-label="`${token.label} RGB color`"
                    @input="setRgb(token.id, ($event.target as HTMLInputElement).value)"
                  >
                  <input
                    :value="textValues[token.id] ?? colorFor(token.id)"
                    class="appearance-hex-input"
                    maxlength="9"
                    spellcheck="false"
                    :aria-label="`${token.label} hexadecimal color`"
                    @input="editHex(token.id, ($event.target as HTMLInputElement).value)"
                    @change="setColor(token.id, ($event.target as HTMLInputElement).value)"
                  >
                  <label class="appearance-alpha">
                    <span>Alpha {{ alphaPercent(colorFor(token.id)) }}%</span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      :value="alphaPercent(colorFor(token.id))"
                      @input="setAlpha(token.id, Number(($event.target as HTMLInputElement).value))"
                    >
                  </label>
                  <div class="appearance-comparison" aria-label="Original, draft, and default swatches">
                    <span :style="swatchStyle(original.colors[token.id] ?? token.defaultValue)" title="Original" />
                    <span :style="swatchStyle(colorFor(token.id))" title="Draft" />
                    <span :style="swatchStyle(token.defaultValue)" title="Default" />
                  </div>
                  <button type="button" :disabled="colorFor(token.id) === token.defaultValue" @click="resetToken(token)">Reset</button>
                </div>
                </article>
              </div>
            </template>
          </details>
          <p v-if="visibleGroups.length === 0" class="appearance-empty">No color tokens match “{{ query }}”.</p>
        </div>
      </div>

      <p v-if="errorMessage || externalError" class="appearance-message error">{{ errorMessage || externalError }}</p>
      <p v-else-if="notice" class="appearance-message">{{ notice }}</p>

      <footer class="dialog-actions appearance-dialog-actions">
        <span>Changes are previewed live. Cancel restores {{ original.name }}.</span>
        <button class="button secondary" type="button" :disabled="saving || fileBusy" @click="emit('cancel')">Cancel</button>
        <button class="button primary" type="button" :disabled="saving || fileBusy" @click="applyTemplate">
          {{ saving ? "Applying…" : "Apply template" }}
        </button>
      </footer>
    </section>
  </div>
</template>
