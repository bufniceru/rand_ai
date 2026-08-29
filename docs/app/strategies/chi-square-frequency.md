# Chi-square Frequency

## Introduction

**Chi-square Frequency** is the production strategy with identifier
`chi_square` and short engine name **Chi²**. It is a default-enabled member of
the **Frequency & Recency** family. The strategy compares every number's
lifetime appearance count with the common frequency expected under a uniform
6-from-49 reference model, then ranks numbers by the signed direction of that
difference.

The engine produces a complete 1–49 ranking. Its first six entries form the
Top-6 prediction used by prediction grids, audits, effectiveness histories,
comparisons, portfolios, exports, and selected ensemble consumers.

```{admonition} Distinct from Categorical Chi-square
:class: important

`chi_square` is the lifetime-frequency strategy documented here. It is not
`categorical_chi_square`, displayed as **Categorical Chi-square**. The latter is
a separate conditional model involving exact gap and left/right-space states.
The two engines share statistical terminology but not their state,
probabilities, or ranking calculation.
```

## Scope

This page documents the exact production count state, uniform expectation,
signed Pearson residual, per-number chi-square contribution, qualitative
residual bands, min–max transformation, causal lifecycle, tie-breaking, and
retrospective walk-forward behavior of `chi_square`.

The strategy uses descriptive frequency deviations. It does not perform a
global chi-square significance test, emit a p-value, infer that an over-frequency
number is “due,” or establish that historical frequency predicts the next draw.

## Prediction problem

After {math}`h` completed draws, let

```{math}
O_n(h)=\sum_{t=1}^{h}\mathbf{1}[n\in D_t]
```

be the lifetime number of appearances of candidate
{math}`n\in\{1,\ldots,49\}`. Each draw contains exactly six unique numbers, so

```{math}
\sum_{n=1}^{49}O_n(h)=6h.
```

Under a uniform 6-from-49 reference model, every number has marginal inclusion
probability

```{math}
p_0=\frac{6}{49}.
```

Its expected lifetime count after {math}`h` draws is therefore

```{math}
E(h)=hp_0=\frac{6h}{49}.
```

The strategy asks only whether each observed count is above or below this same
expectation and by how much. It contains no trainable weights, recent window,
decay factor, gap-frequency table, or conditional state model.

## Core mathematics

### Signed Pearson residual

For {math}`h>0`, production computes

```{math}
r_n=\frac{O_n-E}{\sqrt{E}}.
```

The sign is retained:

- {math}`r_n>0` means the number has appeared more often than the uniform
  expectation;
- {math}`r_n<0` means it has appeared less often;
- {math}`r_n=0` means its count equals the expectation exactly.

When {math}`h=0`, the expected count is zero and production assigns every
residual the guarded fallback value 0 rather than dividing by zero.

This is called a signed Pearson residual because squaring it gives the
number's Pearson chi-square contribution:

```{math}
c_n=\frac{(O_n-E)^2}{E}=r_n^2,
```

with {math}`c_n=0` when {math}`E=0`. The contribution measures magnitude but
loses direction, so it is shown for diagnostics and is not used as the ranking
score. Ranking by {math}`c_n` would favor both unusually low and unusually high
counts; the production strategy instead favors the positive side of
{math}`r_n`.

### Residual bands

Each number receives one exact descriptive band:

| Band | Production condition |
|---|---|
| Strong under | {math}`r_n\leq-2` |
| Mild under | {math}`-2<r_n\leq-1` |
| Near expected | {math}`-1<r_n<1` |
| Mild over | {math}`1\leq r_n<2` |
| Strong over | {math}`r_n\geq2` |

The bands are labels only. They do not change a score, enforce a quota, or
represent hypothesis-test decisions.

## What the strategy actually ranks

At any fixed history length {math}`h>0`, both {math}`E` and {math}`\sqrt E`
are identical for all 49 candidates. Consequently,

```{math}
r_i>r_j \quad\Longleftrightarrow\quad O_i>O_j.
```

The raw strategy ordering is therefore exactly a lifetime “hot-number” ordering:
numbers with more historical appearances rank ahead of numbers with fewer
appearances. The expectation and residual scale make the deviation easier to
interpret, but they do not change that ordering.

This also means the strategy does not favor underrepresented numbers. A large
negative residual is statistically conspicuous in magnitude, but it remains
below a smaller or positive residual in the production ranking.

## Score transformation

Let

```{math}
r_{\min}=\min_n r_n, \qquad r_{\max}=\max_n r_n.
```

When {math}`r_{\max}>r_{\min}`, the application maps every residual to

```{math}
s_n=\frac{r_n-r_{\min}}{r_{\max}-r_{\min}}.
```

The smallest current residual becomes 0 and the largest becomes 1. Because the
residual is a common affine transformation of {math}`O_n`, the normalized score
can equivalently be written as

```{math}
s_n=\frac{O_n-O_{\min}}{O_{\max}-O_{\min}}.
```

Thus the expected count cancels completely from the displayed score whenever
the count range is nonzero. If all residuals are equal, including an empty
history, production assigns all 49 scores the fallback value 0.

The displayed percentage is a relative position inside the current lifetime
count range. It is not the probability that a number will appear in the next
draw, a p-value, or a calibrated measure of evidence.

## Causal walk-forward lifecycle

The strategy shares the application's train-then-remember-then-score replay
contract, although `chi_square` has no separate training update. When draw
{math}`D_t` becomes observed:

1. Any pending prediction for {math}`D_t` is evaluated without changing its
   historical ranking.
2. The six appearance counters for numbers in {math}`D_t` are incremented.
3. The completed-draw count becomes {math}`t`.
4. The expectation {math}`6t/49`, residuals, contributions, bands, and scores
   are calculated for the forecast of {math}`D_{t+1}`.

Features for a target therefore use only draws that precede it. The target draw
cannot increase its own candidate counts. Appending later draws cannot alter a
ranking already produced for an earlier history prefix.

### First forecast

Before any draw is remembered, all raw and normalized scores are zero. In the
normal application lifecycle, however, the first displayed forecast is built
after the first draw has been remembered. Its six observed numbers have count 1
and the other 43 have count 0, so the six just-observed numbers receive score 1
and form the first Top-6. This is a direct consequence of lifetime-frequency
ranking, not a special repeat-draw rule.

## Ranking and tie-breaking

Candidates are ordered by:

1. larger normalized signed-residual score;
2. larger zero-based current gap when scores tie;
3. smaller number when both score and gap tie.

Because min–max scaling is monotone when the residual range is nonzero, the
first rule is equivalent to sorting by appearance count. Current gap affects
only candidates with equal lifetime counts. A number present in the latest
reference draw has current gap 0. The first six candidates after these rules
form the Top-6.

The gap tie-break adds a recency distinction among count ties, but gap is not
part of the chi-square score or residual.

## Interpreting the application fields

| Field | Meaning |
|---|---|
| Score | Min–max normalized signed residual, displayed as a percentage. It is equivalent to the candidate's relative position between the smallest and largest lifetime counts. |
| Residual band | Strong under, Mild under, Near expected, Mild over, or Strong over according to the exact thresholds above. |
| Observed versus expected | Lifetime appearances {math}`O_n` compared with the common value {math}`6h/49`, displayed to two decimal places. |
| Signed Pearson residual | {math}`r_n`, retaining whether the deviation is below or above expectation and displayed to three decimal places. |
| Chi-square contribution | {math}`c_n=r_n^2`, displayed to three decimal places. |
| Rank | Position in the complete 1–49 ordering after score and tie-breaking rules. |
| Top-6 membership | Whether the number occupies one of the first six ranks. |

Two numbers with the same appearance count always have the same residual,
contribution, normalized score, and band. Their relative order comes only from
the current-gap and number tie-breaks.

## Statistical interpretation

### Marginal count distribution

If successive draws are independent and uniformly sample six unique numbers,
the marginal lifetime count for one fixed number is

```{math}
O_n\sim\operatorname{Binomial}\left(h,\frac{6}{49}\right).
```

Its mean is the production expectation {math}`E=6h/49`, while its variance is

```{math}
\operatorname{Var}(O_n)=
h\frac{6}{49}\frac{43}{49}=E\frac{43}{49}.
```

The production residual divides by {math}`\sqrt E`, not by the exact marginal
standard deviation {math}`\sqrt{E(43/49)}`. Under this ideal null, its variance
is therefore

```{math}
\operatorname{Var}(r_n)=\frac{43}{49}\approx0.877551,
```

rather than exactly 1. In addition, the 49 number counts are negatively related
within each draw because a draw cannot contain the same number twice and must
contain exactly six distinct numbers.

### Relation to a global chi-square statistic

Summing the displayed per-number contributions would produce

```{math}
Q=\sum_{n=1}^{49}\frac{(O_n-E)^2}{E}.
```

The production strategy does not compute {math}`Q`, compare it with a
chi-square distribution, assign degrees of freedom, or calculate a p-value.
Treating all {math}`6h` observed balls as independent multinomial trials would
also ignore the without-replacement structure inside each draw. The name
“Chi-square Frequency” refers to the residual and contribution form, not to a
completed significance test.

### Top-6 null baseline

For any six-number prediction evaluated against an independent uniform
six-number draw, the number of hits follows

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6).
```

Its mean and variance are

```{math}
\mathbb{E}[H]=\frac{36}{49}\approx0.734694,
```

```{math}
\operatorname{Var}(H)=
6\frac{6}{49}\frac{43}{49}\frac{43}{48}
\approx0.577572.
```

This overlap distribution is the appropriate theoretical reference for Top-6
hit totals. It does not turn a retrospective comparison into an independent
hypothesis test.

## Leakage-free replay evidence

A source-concordance replay through the production `build_prediction_suites`
path uses the repository's 771 chronological YAML draws and produces 770 target
forecasts. It records:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 605 | 0.785714 | 565.714 |
| Validation, target draws 121–520 | 400 | 331 | 0.827500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 176 | 0.704000 | 183.673 |

The stronger validation result does not persist in the later holdout slice,
where the recorded mean is below the theoretical random expectation. The full
replay total is also influenced by the earlier development interval. These
figures are acceptance and regression evidence for the implementation, not
evidence of a stable future advantage.

No significance claim is made. The strategy is one of many engines inspected
on the same dataset, the slice choices are known, and the lifetime state carries
earlier observations into later forecasts.

## Limitations and responsible interpretation

- **Hot-number assumption:** the ranking always favors larger lifetime counts;
  it does not establish that over-frequency persists.
- **No recent adaptation:** all completed draws receive equal, permanent
  weight. A changed process would be diluted by the full history.
- **No conditional context:** the model ignores draw order beyond cumulative
  counts, co-occurrence, spacing, gaps, and interactions between numbers.
- **Residual approximation:** {math}`\sqrt E` is the Pearson contribution
  denominator, not the exact binomial standard deviation for one number's
  without-replacement inclusion history.
- **Dependent categories:** the 49 counts are constrained to sum to {math}`6h`
  and are not independent within a draw.
- **No global test:** per-number contributions are displayed without a global
  statistic, reference degrees of freedom, multiple-testing correction, or
  p-value.
- **Tie-break influence:** current gap can determine Top-6 membership among
  equal-frequency numbers even though it is absent from the primary score.
- **Min–max information loss:** every nonconstant history is stretched to the
  full 0–1 range, regardless of whether its residual spread is small or large.
- **Uncalibrated percentage:** the displayed score is relative to current count
  extremes and is not an occurrence probability.
- **Dataset and selection dependence:** retrospective results depend on the
  exact history, strategy comparisons, and chosen evaluation slices.
- **No guaranteed predictability:** frequency deviations are expected in random
  samples and may regress, persist by chance, or reverse on future draws.

Use the strategy to inspect and rank signed historical frequency deviations,
not as proof that a fair draw has memory.

## Implementation map

| Responsibility | Production location |
|---|---|
| Constants, completed-draw count, appearance counters, and last-seen state | `src/rand_ai/strategy_prediction.py`, module constants and `_StrategyState.__init__` |
| Causal count updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Expectation, residual, contribution, bands, and min–max score | `src/rand_ai/strategy_prediction.py`, `_StrategyState._chi_square_scores` and `_scale_scores` |
| Full ranking and Top-6 construction | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `_StrategyState.build_strategies` |
| Walk-forward orchestration and efficacy | `src/rand_ai/strategy_prediction.py`, `build_prediction_suites` and `_EfficacyTracker` |
| Desktop bridge serialization | `src/rand_ai/gui_bridge.py`, `_strategy_payload` and `_suite_payload` |
| Full display name, default registration, family, details, and score rendering | `web/electron/main.cjs`, `web/src/lib/strategyFamilies.ts`, `web/src/components/SettingsDialog.vue`, and `web/src/views/CombinedPredictionGridView.vue` |
| Direct residual, band, scaling, and detail regression test | `tests/test_strategy_prediction.py`, `test_chi_square_ranks_signed_frequency_residuals` |
| Ranking shape, Top-6, efficacy, and bridge payload coverage | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Concordance replay dataset | `data/lotto_results_2019.yaml` |

The serializer exposes all 49 normalized scores, ranks, current gaps, detail
strings, Top-6 numbers, and completed efficacy. Presentation code does not
recalculate the residuals or alter the ranking.
