# Nonlinear Dynamics and Recurrence Dynamics

## Purpose

Rand AI includes two related features:

- **Nonlinear Dynamics** is a default-enabled diagnostic report.
- **Recurrence Dynamics (Experimental)** is an opt-in prediction strategy.

They test whether similar historical draw states have similar successors. They do
not assume that lottery draws are chaotic, and recurrence is not evidence that a
fair lottery has become predictable.

## Diagnostic draw state and delay embedding

For recurrence quantification, each chronological draw becomes a 20-value,
order-independent feature vector:

- the six sorted values, normalized to the 1–49 range;
- the six circular empty spaces, normalized by their fixed total of 43;
- normalized sum, span, odd share, low-number share, prime share, and consecutive-pair share;
- overlap with each of the preceding two draws.

The number, space, shape, and overlap blocks are divided by the square root of
their block size so that no block dominates merely by containing more fields.
Three consecutive diagnostic vectors form one 60-value delay embedding with lag
weights `0.50`, `0.75`, and `1.00` from oldest to newest.

## Recurrence forecast V2

The forecast deliberately uses a smaller representation than the diagnostic to
avoid concentrating distances across correlated structural features. Each draw is
represented by its six sorted values, normalized to the 1–49 range and divided by
the square root of six. Three draws form an 18-value embedding with the same lag
weights as the diagnostic.

For the latest forecast state, the strategy:

1. excludes states within three draws of the target state;
2. keeps the nearest 8 eligible historical states;
3. weights an analogue at distance `d` by `1 / (d + 1e-9)`;
4. normalizes the weights to sum to the number of retained analogues;
5. counts each number in the draw following every analogue;
6. shrinks the weighted count with eight draws of uniform `6/49` prior evidence.

The posterior score for number `n` is

```text
(weighted successor hits[n] + 8 × 6/49) / (sum of analogue weights + 8)
```

The normal strategy ranking contract resolves equal scores by current gap and
number. Every historical prediction uses only draws available at that time.

## Evidence labels

The report analyzes at most the latest 750 draws. It builds a recurrence matrix at
a fixed 10% distance threshold, excludes the three-draw temporal neighborhood,
and reports recurrence rate, determinism, diagonal lengths, laminarity, and
trapping time. Determinism is compared with 99 fixed-seed shuffled-order
surrogates.

The forecast is evaluated walk-forward against the theoretical random expectation
of `36/49 = 0.7346938776` Top-6 hits per draw. Strategy metadata reports forecast
skill without claiming that recurrence itself is significant:

| Label | Requirement |
|---|---|
| Insufficient | Fewer than 100 evaluated forecasts or fewer than 8 current analogues |
| Weak | Mean does not exceed random |
| Suggestive | Mean exceeds random but the 95% lower confidence bound does not |
| Supported | The 95% lower confidence bound exceeds random |

The report verdict is stricter. It is weak when the shuffled-order surrogate test
fails, suggestive only when surrogate `p <= 0.05` and mean forecast performance
exceeds random, and supported only when both conditions hold and the forecast's
95% lower bound also exceeds random.

The per-prediction evidence index combines effective-neighbor support relative to
the fixed eight-neighbor maximum and the causal percentile of the current nearest-
neighbor distance. It is a support indicator, not a calibrated probability of
winning.

## Interpretation and limitations

Chaos is deterministic sensitivity to initial conditions. Lottery randomness is
stochastic. Delay embedding and recurrence analysis can detect repeated geometry
in a recorded sequence, but they cannot establish that the lottery mechanism is a
low-dimensional deterministic system. Multiple-testing, finite-history, and
selection effects can also create apparent structure.

For these reasons the prediction strategy is opt-in and permanently labeled
experimental. A favorable result should be replicated on later, untouched draws
before it is treated as anything more than a research observation.

V2 parameters were selected after exploratory work on the included history, so
that history is now regression evidence rather than independent confirmation. See
[the fixed benchmark](../reports/recurrence_dynamics_benchmark.md) for the recorded
V1 and V2 results.

## SVC–Recurrence Hybrid

**SVC–Recurrence Hybrid (Experimental)** is a separate opt-in strategy that
combines the full 1–49 rankings from Recurrence Dynamics and the online Support
Vector Classifier. It does not change either source strategy.

For each source, cumulative Top-6 hits are updated only after the predicted draw
is known. A neutral 24-draw prior shrinks its mean effectiveness toward the
theoretical random expectation:

```text
quality = (past hits + 24 × 36/49) / (evaluated draws + 24)
```

The two qualities are normalized into weights. Each number's source rank becomes
`(49 - rank) / 48`, and the hybrid score is the weighted sum of those two rank
strengths. Before any completed forecast, the weights are exactly 50/50. Raw
model scores are not mixed because the source score scales are not comparable.

The hybrid uses no short-window response, agreement bonus, or fixed source quota.
Its result is auditable through the displayed source weights and ranks. See the
[fixed hybrid benchmark](../reports/svc_recurrence_hybrid_benchmark.md); the mixed
historical result is why the strategy remains experimental.
