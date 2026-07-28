<script setup lang="ts">
import { computed } from "vue";
import { buildCoOccurrenceModel } from "../lib/coOccurrence";
import type { AnalysisPayload } from "../types";

const props = defineProps<{ analysis: AnalysisPayload }>();
const model = computed(() =>
  buildCoOccurrenceModel(props.analysis.analysisHistory),
);
const networkNodes = computed(() => {
  const numbers = new Set<number>();
  for (const edge of model.value.networkEdges) {
    numbers.add(edge.numbers[0]);
    numbers.add(edge.numbers[1]);
  }
  const ordered = [...numbers].sort((left, right) => left - right);
  return ordered.map((number, index) => {
    const angle =
      (index / Math.max(ordered.length, 1)) * Math.PI * 2 - Math.PI / 2;
    const node = model.value.nodes.find(
      (candidate) => candidate.number === number,
    );
    return {
      number,
      x: 380 + Math.cos(angle) * 126,
      y: 155 + Math.sin(angle) * 126,
      size:
        7 +
        ((node?.weightedDegree ?? 0) / model.value.maxWeightedDegree) * 11,
    };
  });
});
const nodeByNumber = computed(
  () => new Map(networkNodes.value.map((node) => [node.number, node])),
);
const lowestEdges = computed(() =>
  [...model.value.edges]
    .sort(
      (left, right) =>
        left.lift - right.lift ||
        left.count - right.count ||
        left.pair.localeCompare(right.pair),
    )
    .slice(0, 16),
);

function decimal(value: number, digits = 2): string {
  return value.toFixed(digits);
}

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function bandColor(bandId: string): string {
  return (
    model.value.bands.find((band) => band.id === bandId)?.color ?? "#7b8798"
  );
}
</script>

<template>
  <section class="workspace-view autocorrelation-view co-occurrence-view">
    <header class="view-header">
      <div>
        <p class="eyebrow">Pair diagnostics</p>
        <h2>Co-occurrence report</h2>
      </div>
      <p>
        Full PyLotto same-draw pair analysis across every draw in the active
        dataset.
      </p>
    </header>

    <aside class="warning-banner">
      This is an experimental association model. A strong historical pair is
      not evidence that either number causes or predicts the other.
    </aside>

    <div class="autocorrelation-facts co-occurrence-facts">
      <article>
        <span>Draws analyzed</span>
        <strong>{{ model.drawCount.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Pair events</span>
        <strong>{{ model.totalPairEvents.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Expected / pair</span>
        <strong>{{ decimal(model.expectedPairCount) }}</strong>
      </article>
      <article>
        <span>Possible pairs</span>
        <strong>{{ model.pairUniverseSize.toLocaleString() }}</strong>
      </article>
      <article>
        <span>Latest reference</span>
        <strong>{{ model.latestProfile.date ?? "n/a" }}</strong>
      </article>
      <article>
        <span>Experimental top 6</span>
        <strong class="co-occurrence-top">
          {{
            model.predictions
              .slice(0, 6)
              .map((prediction) => prediction.number)
              .join(", ")
          }}
        </strong>
      </article>
    </div>

    <article class="autocorrelation-panel autocorrelation-interpretation">
      <h3>Interpretation</h3>
      <p>{{ model.interpretation }}</p>
      <div class="autocorrelation-bands co-occurrence-bands">
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
        <h3>Strongest pair network</h3>
        <svg
          class="autocorrelation-chart co-occurrence-network"
          viewBox="0 0 760 310"
          role="img"
          aria-label="The 36 strongest historical pair relationships"
        >
          <g v-for="edge in model.networkEdges" :key="edge.pair">
            <line
              v-if="
                nodeByNumber.get(edge.numbers[0]) &&
                nodeByNumber.get(edge.numbers[1])
              "
              :x1="nodeByNumber.get(edge.numbers[0])?.x"
              :y1="nodeByNumber.get(edge.numbers[0])?.y"
              :x2="nodeByNumber.get(edge.numbers[1])?.x"
              :y2="nodeByNumber.get(edge.numbers[1])?.y"
              :stroke="bandColor(edge.bandId)"
              :stroke-width="1 + (edge.count / model.maxEdgeCount) * 5"
            />
          </g>
          <g v-for="node in networkNodes" :key="node.number">
            <circle
              :cx="node.x"
              :cy="node.y"
              :r="node.size"
              fill="#ffffff"
              stroke="#173d64"
              stroke-width="1.5"
            />
            <text :x="node.x" :y="node.y + 3.5">{{ node.number }}</text>
          </g>
        </svg>
      </article>

      <article class="autocorrelation-panel">
        <h3>Latest draw pair profile</h3>
        <p class="autocorrelation-signature">
          {{ model.latestProfile.signature }}
        </p>
        <div class="co-occurrence-edge-cards">
          <div
            v-for="edge in model.latestProfile.edges"
            :key="edge.pair"
            :style="{ '--autocorrelation-color': bandColor(edge.bandId) }"
          >
            <strong>{{ edge.pair }}</strong>
            <span>{{ edge.label }}</span>
            <small>
              count {{ edge.count }} · lift {{ decimal(edge.lift) }}
            </small>
          </div>
        </div>
      </article>
    </div>

    <div class="autocorrelation-layout">
      <article class="autocorrelation-panel">
        <h3>Top co-occurring pairs</h3>
        <div class="autocorrelation-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Pair</th>
                <th>Count</th>
                <th>Lift</th>
                <th>Residual</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="edge in model.edges.slice(0, 28)" :key="edge.pair">
                <td>{{ edge.rank }}</td>
                <td>{{ edge.pair }}</td>
                <td>{{ edge.count }}</td>
                <td>{{ decimal(edge.lift) }}</td>
                <td>{{ decimal(edge.residual) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>

      <article class="autocorrelation-panel">
        <h3>Number hubs</h3>
        <div class="autocorrelation-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>No.</th>
                <th>Degree</th>
                <th>Best partner / count</th>
                <th>Lift</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="node in model.nodes.slice(0, 28)" :key="node.number">
                <td>{{ node.rank }}</td>
                <td>{{ node.number }}</td>
                <td>{{ node.weightedDegree }}</td>
                <td>
                  {{ node.strongestPartner ?? "n/a" }} /
                  {{ node.strongestPartnerCount }}
                </td>
                <td>{{ decimal(node.strongestPartnerLift) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </div>

    <div class="autocorrelation-layout">
      <article class="autocorrelation-panel">
        <h3>Next draw co-occurrence ranking</h3>
        <div class="autocorrelation-number-grid co-occurrence-number-grid">
          <div
            v-for="prediction in model.predictions"
            :key="prediction.number"
            :style="{
              '--autocorrelation-color': bandColor(prediction.bandId),
            }"
            :title="`${prediction.label}; score ${decimal(prediction.score)}; average lift ${decimal(prediction.averageLift)}; total pair count ${prediction.totalCount}; strongest partner ${prediction.strongestPartner ?? 'n/a'}`"
          >
            <strong>{{ prediction.number }}</strong>
            <span>#{{ prediction.rank }} · {{ decimal(prediction.score, 1) }}</span>
          </div>
        </div>
      </article>

      <article class="autocorrelation-panel">
        <h3>Top experimental picks</h3>
        <div class="autocorrelation-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>No.</th>
                <th>Score</th>
                <th>Avg lift</th>
                <th>Best partner / count</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="prediction in model.predictions.slice(0, 16)"
                :key="prediction.number"
              >
                <td>{{ prediction.rank }}</td>
                <td>{{ prediction.number }}</td>
                <td>{{ decimal(prediction.score, 1) }}</td>
                <td>{{ decimal(prediction.averageLift) }}</td>
                <td>
                  {{ prediction.strongestPartner ?? "n/a" }} /
                  {{ prediction.strongestPartnerCount }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </div>

    <div class="autocorrelation-layout">
      <article class="autocorrelation-panel">
        <h3>Lift bands</h3>
        <div class="co-occurrence-band-summary">
          <div v-for="band in model.bands" :key="band.id">
            <span>
              <strong>{{ band.label }}</strong>
              <small>
                {{
                  model.edges.filter((edge) => edge.bandId === band.id).length
                }}
                pairs
              </small>
            </span>
            <i>
              <b
                :style="{
                  width: `${Math.max(
                    2,
                    (model.edges.filter((edge) => edge.bandId === band.id)
                      .length /
                      model.pairUniverseSize) *
                      100,
                  )}%`,
                  background: band.color,
                }"
              />
            </i>
            <em>
              {{
                percent(
                  model.edges.filter((edge) => edge.bandId === band.id)
                    .length / model.pairUniverseSize,
                )
              }}
            </em>
          </div>
        </div>
      </article>

      <article class="autocorrelation-panel">
        <h3>Lowest co-occurring pairs</h3>
        <div class="autocorrelation-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Pair</th>
                <th>Count</th>
                <th>Expected</th>
                <th>Lift</th>
                <th>Share</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="edge in lowestEdges" :key="edge.pair">
                <td>{{ edge.pair }}</td>
                <td>{{ edge.count }}</td>
                <td>{{ decimal(edge.expected) }}</td>
                <td>{{ decimal(edge.lift) }}</td>
                <td>{{ percent(edge.share) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </div>
  </section>
</template>
