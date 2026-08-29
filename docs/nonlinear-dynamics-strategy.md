# Nonlinear Dynamics and Recurrence Dynamics

## Purpose

Rand AI includes two related features:

- **Nonlinear Dynamics** is a default-enabled diagnostic report.
- **Recurrence Dynamics (Experimental)** is an opt-in prediction strategy.

They test whether similar historical draw states have similar successors. They do
not assume that lottery draws are chaotic, and recurrence is not evidence that a
fair lottery has become predictable.

## Draw state and delay embedding

Each chronological draw becomes a 20-value, order-independent feature vector:

- the six sorted values, normalized to the 1–49 range;
- the six circular empty spaces, normalized by their fixed total of 43;
- normalized sum, span, odd share, low-number share, prime share, and consecutive-pair share;
- overlap with each of the preceding two draws.

The number, space, shape, and overlap blocks are divided by the square root of
their block size so that no block dominates merely by containing more fields.
Three consecutive vectors form one 60-value delay embedding with lag weights
`0.50`, `0.75`, and `1.00` from oldest to newest.

## Recurrence forecast

For the latest embedded state, the strategy:

1. excludes states within three draws of the target state;
2. keeps the nearest 24 eligible historical states;
3. weights an analogue at distance `d` by `exp(-d / max(d_min, 1e-12))`;
4. rescales the weights to sum to the number of retained analogues;
5. counts each number in the draw following every analogue;
6. shrinks the weighted count with eight draws of uniform `6/49` prior evidence.

The posterior score for number `n` is

```text
(weighted successor hits[n] + 8 × 6/49) / (sum of analogue weights + 8)
```

The normal strategy ranking contract then resolves equal scores by current gap
and number. Every historical prediction uses only draws available at that time.

## Evidence labels

The report analyzes at most the latest 750 draws. It builds a recurrence matrix
at a fixed 10% distance threshold, excludes the three-draw temporal neighborhood,
and reports recurrence rate, determinism, diagonal lengths, laminarity, and
trapping time. Determinism is compared with 99 fixed-seed shuffled-order
surrogates.

The recurrence forecast is also evaluated walk-forward against the theoretical
random expectation of `36/49 = 0.7346938776` Top-6 hits per draw. Labels use the
following fixed policy:

| Label | Requirement |
|---|---|
| Insufficient | Fewer than 100 evaluated forecasts or fewer than 8 current analogues |
| Weak | Mean does not exceed random, or the surrogate p-value is above 0.05 |
| Suggestive | Surrogate p-value is at most 0.05 and the mean exceeds random |
| Supported | Suggestive conditions plus a 95% lower confidence bound above random |

The per-prediction evidence index combines effective-neighbor support and the
causal percentile of the current nearest-neighbor distance. It is a support
indicator, not a calibrated probability of winning.

## Interpretation and limitations

Chaos is deterministic sensitivity to initial conditions. Lottery randomness is
stochastic. Delay embedding and recurrence analysis can detect repeated geometry
in a recorded sequence, but they cannot establish that the lottery mechanism is a
low-dimensional deterministic system. Multiple-testing, finite-history, and
selection effects can also create apparent structure.

For these reasons the prediction strategy is opt-in and permanently labeled
experimental. A favorable result should be replicated on later, untouched draws
before it is treated as anything more than a research observation.

See [the fixed benchmark](../reports/recurrence_dynamics_benchmark.md) for the
current dataset result.
