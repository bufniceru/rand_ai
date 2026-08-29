# Bayesian

## Introduction

**Bayesian** is the production strategy with identifier `bayesian` and short
engine name **Baye**. It is a default-enabled member of the
**Frequency & Recency** family. The strategy estimates occurrence rates from a
number's current recurrence-gap bucket, a slowly decayed gap history, and a
faster number-specific recency history. Three prior-smoothed estimates are
combined into one score for each number from 1 through 49.

The first six ranks form its Top-6 prediction for grids, audits, effectiveness
histories, comparisons, portfolios, exports, and Possible Draw. Its complete
ranking is also used by several optional ensemble consumers, but those
consumers do not change the standalone calculation documented here.

```{admonition} Distinct from Border Group Bayesian
:class: important

`bayesian` is the gap-and-number strategy on this page. It is not
`border_group_bayesian`, which forecasts circular group signatures from
categorical context. The two engines have different state, mathematics, and
outputs.
```

## Scope

The strategy asks:

> Given a candidate's current gap, the historical hit rate of that gap bucket,
> and the candidate's own decayed recent activity, what smoothed occurrence
> rate should be assigned to the next draw?

The answer is an engineered model average of three posterior means:

- 50% lifetime gap evidence;
- 15% slowly decayed recent gap evidence;
- 35% faster number-specific recent evidence.

The model does not produce a joint distribution over valid six-number tickets.
Its 49 marginal scores are ranked independently, and the displayed percentages
are not calibrated guarantees.

## Prediction problem

For each completed history and each candidate \(n\in\{1,\ldots,49\}\), define

\[
y_t(n)=\mathbf 1[n\in D_t],
\]

where \(D_t\) is the six-number draw. The common random-reference probability
is

\[
p_0=P(y_t(n)=1)=\frac{6}{49}=0.12244898\ldots.
\]

Each draw supplies exactly 49 candidate exposures and six positive outcomes.
This class imbalance is handled through strong shrinkage toward \(p_0\), not by
reweighting positive observations.

## Recurrence-gap state

The gap used to predict draw \(t\) is computed before \(D_t\) is known. If
number \(n\) appeared in the immediately preceding draw, its gap is 0. Every
intervening completed draw in which it did not appear increases the gap by 1.
For a number not yet seen in the available prefix, the gap equals the number of
completed draws.

The production bucket is

\[
b_t(n)=\min(g_t(n),35).
\]

There are therefore 36 buckets, indexed 0 through 35. Bucket 35 pools every
gap of 35 or more. The exact uncapped current gap is retained by the application
and can still participate in the final ranking tie-break.

For every target after the first historical draw, all 49 candidates contribute
one opportunity to their pre-target bucket. A drawn candidate also contributes
one hit to that bucket. Let

\[
O_b=\sum_{t,n}\mathbf 1[b_t(n)=b],
\qquad
H_b=\sum_{t,n}\mathbf 1[b_t(n)=b]y_t(n).
\]

These lifetime counters pool candidates that share a gap state; they are not
specific to one number.

## Bayesian prior interpretation

For a Bernoulli hit probability \(p\), a Beta prior with total strength
\(\kappa\) and mean \(p_0\) can be written

\[
p\sim\operatorname{Beta}(\kappa p_0,\kappa(1-p_0)).
\]

After \(H\) hits in \(O\) opportunities, its posterior mean is

\[
\mathbb E[p\mid H,O]
=\frac{H+\kappa p_0}{O+\kappa}.
\]

Bayesian uses this posterior-mean form with two fixed strengths:

| Evidence family | Prior strength | Prior mean |
|---|---:|---:|
| Gap-bucket evidence | 1024 candidate opportunities | \(6/49\) |
| Number-specific recent evidence | 64 effective draws | \(6/49\) |

The large gap prior strongly stabilizes sparse buckets. “Strength 1024” means
1024 candidate opportunities for each gap bucket, not 1024 complete draws.

```{admonition} Bayesian approximation
:class: note

The lifetime component has the ordinary Beta–Bernoulli posterior-mean form.
The recency components use exponentially discounted fractional counts and the
final result is a fixed model average. This is Bayesian-style shrinkage, not a
single fitted generative posterior over the complete lottery process.
```

## Lifetime gap posterior

The lifetime estimate for bucket \(b\) is

\[
P^{\mathrm{life}}_b
=\frac{H_b+1024p_0}{O_b+1024}.
\]

Every number currently assigned to bucket \(b\) receives the same lifetime gap
component. The counts never decay, so this component changes progressively more
slowly as history grows.

The first historical draw is not entered into the gap tables because no prior
draw exists from which to define a normal predictive exposure. Gap evidence
begins with the second draw.

## Slowly decayed recent gap posterior

Recent gap evidence uses the per-draw decay factor

\[
\delta_g=2^{-1/1000}.
\]

Before the current draw's evidence is added, every recent opportunity and hit
counter is multiplied by \(\delta_g\):

\[
\widetilde O_b\leftarrow\delta_g\widetilde O_b,
\qquad
\widetilde H_b\leftarrow\delta_g\widetilde H_b.
\]

The current 49 opportunities and six observed hits are then added to their
pre-draw buckets. Evidence 1000 updates old therefore retains half its original
weight. The posterior mean is

\[
P^{\mathrm{recent\ gap}}_b
=\frac{\widetilde H_b+1024p_0}
       {\widetilde O_b+1024}.
\]

This component shares the same strong gap prior as the lifetime component. Its
1000-draw half-life is deliberately slow and does not represent a fixed
1000-draw window.

## Faster number-specific posterior

For each number \(n\), let \(R_n\) be its exponentially decayed hit mass. The
factor is

\[
\delta_n=2^{-1/100},
\]

and each draw updates

\[
R_n\leftarrow\delta_n R_n+y_t(n).
\]

Because every valid draw contributes six hits under the same decay, the common
effective number of draws is recovered as

\[
E=\frac{1}{6}\sum_{n=1}^{49}R_n.
\]

The number-specific posterior mean is

\[
P^{\mathrm{recent\ number}}_n
=\frac{R_n+64p_0}{E+64}.
\]

Evidence 100 draws old retains half weight. Unlike the two gap components,
this term distinguishes numbers within the same gap bucket according to their
own recent activity.

## Model averaging

For candidate \(n\) with current bucket \(b=b_t(n)\), the unscaled production
estimate is

\[
P_n
=0.50P^{\mathrm{life}}_b
+0.15P^{\mathrm{recent\ gap}}_b
+0.35P^{\mathrm{recent\ number}}_n.
\]

The weights are fixed. They do not adapt to recent efficacy, estimate
uncertainty, or target-draw outcomes.

The three components are correlated because they reuse overlapping draws.
Consequently, \(P_n\) is best interpreted as a model-averaged ranking value,
not as a posterior derived from three independent sources.

## Score normalization and ranking

The 49 unscaled estimates are min–max normalized:

\[
S(n)=
\begin{cases}
\dfrac{P_n-P_{\min}}{P_{\max}-P_{\min}},
&P_{\max}>P_{\min},\\[8pt]
0,&P_{\max}=P_{\min}.
\end{cases}
\]

Numbers are ranked by:

1. larger normalized score;
2. larger exact current gap;
3. smaller number.

The first six ranks form the Top-6 prediction. Min–max normalization preserves
unequal-score order but removes the absolute probability level and spread. A
displayed score of 100% means the largest model average in that draw, not a
100% chance of occurrence.

### Cold start

With no evidence, every component equals \(p_0\). All unscaled estimates are
equal, all normalized scores are zero, and the standard gap/number tie-break
determines the ranking.

The first observed draw is included immediately in the number-specific recent
tracker but not the gap tables. The first emitted next-draw prediction can
therefore distinguish its six observed numbers through the 35% recent-number
component while both gap components remain at their common prior.

## Causal lifecycle

For target draw \(t\), the production sequence is:

1. A prediction for \(t\) already exists from history ending at \(t-1\).
2. When \(D_t\) occurs, decay the recent gap and number evidence before adding
   new observations.
3. For every candidate, calculate its gap bucket using only history through
   \(t-1\), then increment lifetime and recent gap opportunities.
4. Increment gap hits and number-specific hit mass only for the six members of
   \(D_t\).
5. Remember their new last-seen positions and advance the completed-draw count.
6. Calculate current gaps and the three posterior components for target
   \(t+1\), then normalize and rank all 49 candidates.

The observed target is never entered before its own forecast. It affects only
later predictions. Replaying a fixed prefix therefore produces the same final
prefix ranking whether or not additional future draws are appended.

## Interpreting prediction details

Every number displays six detail lines:

- **Model-averaged probability** is the unscaled \(P_n\), formatted as a
  percentage. It is a ranking estimate, not a demonstrated calibrated
  probability.
- **Gap bucket** is the capped state \(\min(g,35)\).
- **Lifetime gap** reports \(P^{\mathrm{life}}_b\) and its 50% blend weight.
- **Recent gap** reports \(P^{\mathrm{recent\ gap}}_b\), its 15% weight, and
  the 1000-draw half-life.
- **Recent number** reports \(P^{\mathrm{recent\ number}}_n\), its 35% weight,
  and the 100-draw half-life.
- **Hierarchical prior strengths** reports 1024 for gap evidence and 64 for
  number evidence.

Rank, current gap, Top-6 membership, and normalized score are supplied by the
standard strategy payload. The detail probability can differ only slightly
between numbers even when min–max scaling displays a large score separation.

## Endpoint evidence scale

After processing all 771 repository draws, the production state contains:

| Evidence | Effective exposure | Effective hits |
|---|---:|---:|
| Lifetime gap tables | 37,730 candidate opportunities across 770 evaluated draws | 4,620 |
| Recent gap tables | 596.880 effective draws | 3,581.282 weighted hits |
| Recent number tracker | 144.079 effective draws | 864.472 weighted hits |

The recent totals are fractional because of exponential decay. Their aggregate
hit rates are mechanically tied to six hits among 49 candidates per valid draw;
they are state-size diagnostics, not evidence that the strategy predicts better
than chance.

## Top-6 efficacy reference

For a fixed six-number prediction evaluated against a uniformly random
six-number draw,

\[
H\sim\operatorname{Hypergeometric}(49,6,6),
\]

with

\[
\mathbb E[H]=\frac{36}{49}=0.734694,
\qquad
\operatorname{Var}(H)=0.577572.
\]

A leakage-free replay over the repository's 771 chronological YAML draws
produces 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 596 | 0.774026 | 565.714 |
| Validation, target draws 121–520 | 400 | 315 | 0.787500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 186 | 0.744000 | 183.673 |

The full replay is above the theoretical random mean, while the holdout excess
is small. These results are retrospective implementation measurements. The
model, weights, priors, and broader strategy collection were developed with
historical data available, so no statistical significance, stable future lift,
or guaranteed predictability is claimed.

## Core mathematical and statistical concepts

- **Bernoulli candidate outcomes:** every number is treated as hit or miss for
  each target draw.
- **Gap-state pooling:** candidates with the same capped recurrence gap share a
  common estimated hit rate.
- **Beta shrinkage:** posterior means pull sparse gap and number evidence toward
  the uniform \(6/49\) reference.
- **Exponential forgetting:** fractional counts retain all history with
  geometrically decreasing influence rather than a hard cutoff.
- **Half-life parameterization:** the gap and number decays retain half weight
  after 1000 and 100 updates respectively.
- **Effective sample size:** total decayed number hit mass divided by six
  recovers the common discounted exposure count.
- **Fixed model averaging:** three correlated posterior means are combined with
  weights 50%, 15%, and 35%.
- **Min–max normalization:** raw model averages are transformed into relative
  within-draw strengths before ranking.
- **Hypergeometric overlap:** standard efficacy uses the null distribution for
  overlap between two six-element subsets of 49.

## Limitations and responsible interpretation

- **No causal mechanism:** recurrence gaps and recent appearances do not cause
  future outcomes in a fair lottery.
- **Candidate dependence:** exactly six of 49 labels are positive in every
  draw, so the 49 Bernoulli exposures are not independent as a joint system.
- **Pooled gap assumption:** all numbers in one bucket are treated as
  exchangeable by both gap components.
- **Capped tail:** every gap of 35 or more shares one evidence pool, potentially
  hiding differences within long absences.
- **Strong fixed priors:** strength 1024 can dominate sparse gap buckets, while
  strength 64 materially stabilizes number recency.
- **Discounted-posterior approximation:** fractional recency counts are an
  engineered power-likelihood update, not a standard stationary Beta posterior.
- **Overlapping evidence:** lifetime gap, recent gap, and recent number terms
  reuse observations and should not be interpreted as independent confirmation.
- **Fixed retrospective weights:** 50/15/35 is not learned causally from
  untouched future efficacy.
- **Hot-number tendency:** the faster number component can reward a recent
  random streak.
- **First-draw asymmetry:** number recency learns from the first draw while gap
  evidence begins only when a prior gap state exists.
- **No uncertainty output:** the UI shows posterior means but not credible
  intervals, posterior variance, or calibration error.
- **Min–max information loss:** normalized scores can exaggerate small raw
  differences and are not comparable across draws.
- **Tie-break influence:** exact current gap and number can decide equal-score
  rankings outside the Bayesian model average.
- **Dataset and selection dependence:** retrospective slice performance can be
  affected by chance and repeated strategy comparison.
- **No guaranteed predictability:** historical lift can disappear or reverse on
  untouched draws.

Use Bayesian as a transparent, strongly regularized gap-and-recency ranking,
not as evidence that a fair draw has a learnable posterior dependence on its
history.

## Implementation map

| Responsibility | Production location |
|---|---|
| Base rate, bucket cap, prior strengths, half-lives, decay factors, and blend weights | `src/rand_ai/strategy_prediction.py`, `_BASE_PROBABILITY`, `_MAX_GAP_BUCKET`, and `_BAYESIAN_*` constants |
| Lifetime and decayed gap arrays plus number-specific recent state | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Pre-draw gap definition and post-draw current gaps | `src/rand_ai/strategy_prediction.py`, `_gap_before_current_draw` and `current_gaps` |
| Decay-before-learning, opportunities, and hit updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` |
| Last-seen positions and completed-draw count | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Three posterior means, fixed model average, and displayed details | `src/rand_ai/strategy_prediction.py`, `_gap_model_scores` with `weighted=False` |
| Min–max scaling, gap/number tie-break, Top-6, and efficacy | `src/rand_ai/strategy_prediction.py`, `_scale_scores`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop registration, serialization, and default-enabled state | `web/electron/main.cjs` and `src/rand_ai/gui_bridge.py` |
| Settings description, family, color, names, and score details | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Posterior blend, detail fields, decay order, and effective-evidence tests | `tests/test_strategy_prediction.py` |
| Strategy identifier, registration, and serialization coverage | `tests/test_gui_bridge.py`, `web/src/types.ts`, and `web/src/lib/strategyFamilies.test.ts` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 normalized scores, ranks, exact current
gaps, detail strings, Top-6 numbers, and standard completed efficacy. It does
not serialize the internal lifetime and decayed bucket arrays as separate
public fields.
