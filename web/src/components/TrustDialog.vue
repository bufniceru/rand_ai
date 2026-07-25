<script setup lang="ts">
import type { DatasetSelection } from "../types";

defineProps<{
  dataset: DatasetSelection;
}>();

defineEmits<{
  cancel: [];
  confirm: [];
}>();
</script>

<template>
  <div class="modal-backdrop" role="presentation" @click.self="$emit('cancel')">
    <section class="trust-dialog" role="dialog" aria-modal="true" aria-labelledby="trust-title">
      <p class="eyebrow">Security confirmation</p>
      <h2 id="trust-title">Trust this pickle file?</h2>
      <p>
        Pickle files can execute Python code while loading. Continue only if you
        created this file or otherwise fully trust its source.
      </p>
      <dl>
        <div><dt>File</dt><dd>{{ dataset.name }}</dd></div>
        <div><dt>Size</dt><dd>{{ (dataset.sizeBytes / 1024).toFixed(1) }} KiB</dd></div>
        <div><dt>Location</dt><dd :title="dataset.path">{{ dataset.path }}</dd></div>
      </dl>
      <div class="dialog-actions">
        <button class="button secondary" type="button" @click="$emit('cancel')">Cancel</button>
        <button class="button danger" type="button" @click="$emit('confirm')">
          I trust this file — analyze
        </button>
      </div>
    </section>
  </div>
</template>
