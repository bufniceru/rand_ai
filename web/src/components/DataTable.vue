<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { TablePayload, TableValue } from "../types";

const props = withDefaults(
  defineProps<{
    table: TablePayload;
    pageSize?: number;
    searchable?: boolean;
  }>(),
  {
    pageSize: 100,
    searchable: true,
  },
);

const query = ref("");
const page = ref(1);

const filteredRows = computed(() => {
  const normalized = query.value.trim().toLowerCase();
  if (!normalized) return props.table.rows;
  return props.table.rows.filter((row) =>
    props.table.columns.some((column) =>
      String(row[column] ?? "").toLowerCase().includes(normalized),
    ),
  );
});
const pageCount = computed(() =>
  Math.max(1, Math.ceil(filteredRows.value.length / props.pageSize)),
);
const visibleRows = computed(() => {
  const start = (page.value - 1) * props.pageSize;
  return filteredRows.value.slice(start, start + props.pageSize);
});

watch([query, () => props.table], () => {
  page.value = 1;
});
watch(pageCount, (count) => {
  page.value = Math.min(page.value, count);
});

function formatValue(value: TableValue): string {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }
  return value;
}
</script>

<template>
  <section class="data-table-shell">
    <div v-if="searchable" class="table-tools">
      <label>
        <span>Filter table</span>
        <input v-model="query" type="search" placeholder="Search visible values">
      </label>
      <span>{{ filteredRows.length.toLocaleString() }} rows</span>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th v-for="column in table.columns" :key="column">{{ column }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, rowIndex) in visibleRows" :key="rowIndex">
            <td v-for="column in table.columns" :key="column">
              {{ formatValue(row[column]) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="pagination">
      <button :disabled="page === 1" type="button" @click="page -= 1">Previous</button>
      <span>Page {{ page }} of {{ pageCount }}</span>
      <button :disabled="page === pageCount" type="button" @click="page += 1">Next</button>
    </div>
  </section>
</template>
