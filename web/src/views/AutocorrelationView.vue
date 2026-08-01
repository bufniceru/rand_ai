<script setup lang="ts">
import { computed } from "vue";
import { buildAutocorrelationModel } from "../lib/autocorrelation";
import type {
  AnalysisPayload,
  AutocorrelationLagSummary,
} from "../types";

const props = defineProps<{
  analysis: AnalysisPayload;
}>();

const model = computed(() =>
  buildAutocorrelationModel(props.analysis.analysisHistory),
);
const chartScale = computed(() =>
  Math.max(
    ...model.value.lagSummaries.map((summary) => Math.abs(summary.score)),
    0.1,
  ),
);
const topPositiveOverlap = computed(() =>
  [...model.value.lagSummaries]
    .sort(
      (left, right) =>
        right.overlapDelta - left.overlapDelta || left.lag - right.lag,
    )
    .slice(0, 8),
);
const topNegativeOverlap = computed(() =>
  [...model.value.lagSummaries]
    .sort(
      (left, right) =>
        left.overlapDelta - right.overlapDelta || left.lag - right.lag,
    )
    .slice(0, 8),
);
const topPositiveDoublets = computed(() =>
  [...model.value.lagSummaries]
    .sort(
      (left, right) =>
        right.doubletDelta - left.doubletDelta || left.lag - right.lag,
    )
    .slice(0, 8),
);
const topNegativeDoublets = computed(() =>
  [...model.value.lagSummaries]
    .sort(
      (left, right) =>
        left.doubletDelta - right.doubletDelta || left.lag - right.lag,
    )
    .slice(0, 8),
);
const topPositiveTriplets = computed(() =>
  [...model.value.lagSummaries]
    .sort(
      (left, right) =>
        right.tripletDelta - left.tripletDelta || left.lag - right.lag,
    )
    .slice(0, 8),
);
const topNegativeTriplets = computed(() =>
  [...model.value.lagSummaries]
    .sort(
      (left, right) =>
        left.tripletDelta - right.tripletDelta || left.lag - right.lag,
    )
    .slice(0, 8),
);

function decimal(value: number, digits = 3): string {
  return value.toFixed(digits);
}

function signed(value: number, digits = 3): string {
  return `${value >= 0 ? "+" : ""}${decimal(value, digits)}`;
}

function bandColor(bandId: string): string {
  return (
    model.value.bands.find((band) => band.id === bandId)?.color ?? "#727072"
  );
}

function strongestSignal(summary: AutocorrelationLagSummary): string {
  const signals = [
    { label: "number", value: summary.numberPresenceCorrelation },
    { label: "sum", value: summary.sumCorrelation },
    { label: "odd", value: summary.oddCountCorrelation },
    { label: "low", value: summary.lowCountCorrelation },
  ].sort((left, right) => Math.abs(right.value) - Math.abs(left.value));
  const strongest = signals[0]!;
  return `${strongest.label} ${signed(strongest.value)}`;
}
</script>

<template>
  <section class="workspace-view autocorrelation-view">
    <header class="view-header">
      <div>
        <p class="eyebrow">Serial diagnostics</p>
        <h2>Autocorrelation report</h2>
      </div>
      <p>
        Full PyLotto lag analysis across every draw in the active dataset.
      </p>
    </header>

    <aside class="warning-banner">
      Autocorrelation describes repeated historical structure. It does not make
      a future lottery result predictable.
    </aside>

    <div class="autocorrelation-facts">
      <article>
        <span>Draws analyzed</span>
        <strong>{{ model.drawCount.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Maximum lag</span>
        <strong>{{ model.maxLag }}</strong>
      </article>
      <article>
        <span>Strongest lag</span>
        <strong>{{ model.strongestLag?.lag ?? "n/a" }}</strong>
      </article>
      <article>
        <span>Expected overlap</span>
        <strong>{{ decimal(model.expectedOverlap, 3) }}</strong>
      </article>
      <article>
        <span>Expected pairs</span>
        <strong>{{ decimal(model.expectedDoublets, 3) }}</strong>
      </article>
      <article>
        <span>Expected triplets</span>
        <strong>{{ decimal(model.expectedTriplets, 3) }}</strong>
      </article>
    </div>

    <article class="autocorrelation-panel autocorrelation-interpretation">
      <h3>Interpretation</h3>
      <p>{{ model.interpretation }}</p>
      <div class="autocorrelation-bands">
        <div
          v-for="band in model.bands"
          :key="band.id"
          :style="{ '--autocorrelation-color': band.color }"
        >
          <i aria-hidden="true" />
          <span>
            <strong>{{ band.label }}</strong>
            <small>{{ band.description }}</small>
          </span>
        </div>
      </div>
    </article>

    <div class="autocorrelation-layout">
      <article class="autocorrelation-panel">
        <h3>Lag strength</h3>
        <svg
          class="autocorrelation-chart"
          viewBox="0 0 760 320"
          role="img"
          aria-label="Autocorrelation strength for lags 1 through 24"
        >
          <line x1="44" x2="732" y1="260" y2="260" />
          <g
            v-for="(summary, index) in model.lagSummaries"
            :key="summary.lag"
            :transform="`translate(${54 + index * 28}, 0)`"
          >
            <rect
              :height="Math.max(2, (summary.score / chartScale) * 190)"
              width="18"
              x="0"
              :y="260 - Math.max(2, (summary.score / chartScale) * 190)"
              :style="{ fill: bandColor(summary.bandId) }"
              rx="5"
            />
            <text x="9" y="284">{{ summary.lag }}</text>
          </g>
        </svg>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr>
                <th>Lag</th>
                <th>Overlap</th>
                <th>Number r</th>
                <th>Sum r</th>
                <th>Strongest signal</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="summary in model.lagSummaries.slice(0, 12)"
                :key="`lag-${summary.lag}`"
              >
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageOverlap, 2) }}</td>
                <td>{{ signed(summary.numberPresenceCorrelation) }}</td>
                <td>{{ signed(summary.sumCorrelation) }}</td>
                <td>{{ strongestSignal(summary) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>

      <article class="autocorrelation-panel">
        <h3>Latest draw autocorrelation profile</h3>
        <p class="autocorrelation-signature">
          {{ model.latestProfile.date ?? "Date unavailable" }} ·
          {{ model.latestProfile.signature }}
        </p>
        <div class="autocorrelation-latest-numbers">
          <div
            v-for="summary in model.latestProfile.numbers"
            :key="summary.number"
            :style="{ '--autocorrelation-color': bandColor(summary.bandId) }"
          >
            <strong>{{ summary.number }}</strong>
            <span>{{ summary.label }}</span>
            <small>
              lag {{ summary.strongestLag }} ·
              r {{ signed(summary.strongestCorrelation) }}
            </small>
          </div>
        </div>
        <dl class="autocorrelation-extremes">
          <div>
            <dt>Strongest positive presence lag</dt>
            <dd>
              {{ model.strongestPositiveLag?.lag ?? "n/a" }}
              <small>
                {{
                  model.strongestPositiveLag
                    ? signed(
                        model.strongestPositiveLag.numberPresenceCorrelation,
                      )
                    : ""
                }}
              </small>
            </dd>
          </div>
          <div>
            <dt>Strongest negative presence lag</dt>
            <dd>
              {{ model.strongestNegativeLag?.lag ?? "n/a" }}
              <small>
                {{
                  model.strongestNegativeLag
                    ? signed(
                        model.strongestNegativeLag.numberPresenceCorrelation,
                      )
                    : ""
                }}
              </small>
            </dd>
          </div>
        </dl>
      </article>
    </div>

    <div class="autocorrelation-layout">
      <article class="autocorrelation-panel">
        <h3>All lag metrics</h3>
        <div class="autocorrelation-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Lag</th>
                <th>Pairs</th>
                <th>Overlap Δ</th>
                <th>Doublet Δ</th>
                <th>Triplet Δ</th>
                <th>Presence r</th>
                <th>Sum r</th>
                <th>Odd r</th>
                <th>Low r</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="summary in model.lagSummaries"
                :key="`all-lag-${summary.lag}`"
              >
                <td>{{ summary.lag }}</td>
                <td>{{ summary.pairCount }}</td>
                <td>{{ signed(summary.overlapDelta, 2) }}</td>
                <td>{{ signed(summary.doubletDelta) }}</td>
                <td>{{ signed(summary.tripletDelta) }}</td>
                <td>{{ signed(summary.numberPresenceCorrelation) }}</td>
                <td>{{ signed(summary.sumCorrelation) }}</td>
                <td>{{ signed(summary.oddCountCorrelation) }}</td>
                <td>{{ signed(summary.lowCountCorrelation) }}</td>
                <td>{{ decimal(summary.score) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>

      <article class="autocorrelation-panel">
        <h3>Number strongest lags</h3>
        <div class="autocorrelation-number-grid">
          <div
            v-for="summary in model.numberSummaries"
            :key="summary.number"
            :style="{ '--autocorrelation-color': bandColor(summary.bandId) }"
            :title="`${summary.label}; strongest lag ${summary.strongestLag}; r ${signed(summary.strongestCorrelation)}; appearances ${summary.appearances}`"
          >
            <strong>{{ summary.number }}</strong>
            <span>{{ summary.strongestLag }}</span>
          </div>
        </div>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Number</th>
                <th>Lag</th>
                <th>r</th>
                <th>Appearances</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="summary in model.numberSummaries.slice(0, 12)"
                :key="`number-${summary.number}`"
              >
                <td>{{ summary.rank }}</td>
                <td>{{ summary.number }}</td>
                <td>{{ summary.strongestLag }}</td>
                <td>{{ signed(summary.strongestCorrelation) }}</td>
                <td>{{ summary.appearances }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </div>

    <div class="autocorrelation-ranking-grid">
      <article class="autocorrelation-panel">
        <h3>Highest overlap lags</h3>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr><th>Lag</th><th>Overlap</th><th>Delta</th><th>Rate</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in topPositiveOverlap" :key="`oh-${summary.lag}`">
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageOverlap, 2) }}</td>
                <td>{{ signed(summary.overlapDelta, 2) }}</td>
                <td>{{ decimal(summary.overlapRate * 100, 1) }}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
      <article class="autocorrelation-panel">
        <h3>Lowest overlap lags</h3>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr><th>Lag</th><th>Overlap</th><th>Delta</th><th>Rate</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in topNegativeOverlap" :key="`ol-${summary.lag}`">
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageOverlap, 2) }}</td>
                <td>{{ signed(summary.overlapDelta, 2) }}</td>
                <td>{{ decimal(summary.overlapRate * 100, 1) }}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
      <article class="autocorrelation-panel">
        <h3>Highest pair-repeat lags</h3>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr><th>Lag</th><th>Pairs</th><th>Delta</th><th>Expected</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in topPositiveDoublets" :key="`dh-${summary.lag}`">
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageDoublets) }}</td>
                <td>{{ signed(summary.doubletDelta) }}</td>
                <td>{{ decimal(summary.expectedDoublets) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
      <article class="autocorrelation-panel">
        <h3>Lowest pair-repeat lags</h3>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr><th>Lag</th><th>Pairs</th><th>Delta</th><th>Expected</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in topNegativeDoublets" :key="`dl-${summary.lag}`">
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageDoublets) }}</td>
                <td>{{ signed(summary.doubletDelta) }}</td>
                <td>{{ decimal(summary.expectedDoublets) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
      <article class="autocorrelation-panel">
        <h3>Highest triplet-repeat lags</h3>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr><th>Lag</th><th>Triplets</th><th>Delta</th><th>Expected</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in topPositiveTriplets" :key="`th-${summary.lag}`">
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageTriplets) }}</td>
                <td>{{ signed(summary.tripletDelta) }}</td>
                <td>{{ decimal(summary.expectedTriplets) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
      <article class="autocorrelation-panel">
        <h3>Lowest triplet-repeat lags</h3>
        <div class="autocorrelation-table-wrap compact">
          <table>
            <thead>
              <tr><th>Lag</th><th>Triplets</th><th>Delta</th><th>Expected</th></tr>
            </thead>
            <tbody>
              <tr v-for="summary in topNegativeTriplets" :key="`tl-${summary.lag}`">
                <td>{{ summary.lag }}</td>
                <td>{{ decimal(summary.averageTriplets) }}</td>
                <td>{{ signed(summary.tripletDelta) }}</td>
                <td>{{ decimal(summary.expectedTriplets) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </div>
  </section>
</template>
