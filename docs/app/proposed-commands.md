(proposed-command-catalog)=
# Proposed commands catalog

This catalog is the ordered implementation backlog for the Rand AI command
palette. Current executable behavior is documented in {ref}`command-system`.
Entries here describe commands that can be implemented one at a time without
redesigning the command system for each domain.

The order is intentional: descriptive statistics precede conditional models,
sequence models precede their diagnostics, and model diagnostics precede
application actions. Reusing a workspace calculation does not make a command
executable until its registry, bridge, result, and tests have been added.

```{admonition} Interpretation limit
:class: caution

These commands describe historical data and retrospective model behavior.
They do not demonstrate that lottery draws are predictable or that a
historical deviation will continue.
```

## Specification conventions

Every numbered entry is a complete proposal. **Metadata** gives its status,
identifier, category, title, and search keywords. The other lines define its
question, calculation, parameters, result, consumers, dependencies, failure
behavior, and minimum acceptance test.

- **Implemented** is available now.
- **Proposed: existing capability** exposes a calculation or action already
  present elsewhere in Rand AI.
- **Proposed: new capability** needs a new calculation or workflow.
- **Dependent** needs one or more earlier catalog commands first.

Unless stated otherwise, a statistics command reloads the complete active
database, recalculates without prediction preparation or cached Statistics
results, and does not persist its result. Its educational full-screen result
contains a definition, interpretation, limitations, strategy associations,
and the appropriate chart and/or table. **Esc** is the only dismissal method.
Empty datasets disable the command with **Analyze a dataset first**.
Insufficient history is explained in the result; invalid parameters and
calculation failures remain in the same Esc-only result.

Parameters use keyboard-driven quick-picks. Current application settings are
preselected when relevant. Result provenance includes the dataset, complete
draw count, parameters, calculation time, and associated strategies.
Read-only commands execute after their last quick-pick. Write and destructive
commands show a preview and require explicit confirmation.

## 0. Implemented commands

### 1. Statistics: Number Frequency

- **Metadata:** **Implemented**; `statistics.number-frequency`;
  **Statistics**; keywords `number`, `frequency`, `count`, `expected`,
  `residual`.
- **Question/calculation:** How often did each number occur? Count 1--49 and
  compare with {math}`E=N\times6/49`, including rates, deviation and residual.
- **Parameters/result:** None; observed bars and expected line backed by all
  49 rows.
- **Consumers:** frequency, chi-square, entropy, Bayesian, SVC, temporal and
  score-grid strategies; Numbers and Randomness reports.
- **Behavior/acceptance:** See {ref}`command-system`; 49 ordered counts total
  {math}`6N`.

### 2. Statistics: Group Frequency

- **Metadata:** **Implemented**; `statistics.group-frequency`;
  **Statistics**; keywords `group`, `border`, `signature`, `frequency`,
  `count`.
- **Question/calculation:** How often did each canonical border-group signature
  occur? Classify every draw under the selected Border space into one of the
  11 partitions of six.
- **Parameters/result:** Border space defaults to Settings; grouped bars by
  group count and signature, including zero-count rows.
- **Consumers:** every Border Group strategy and Space Groups analysis.
- **Behavior/acceptance:** See {ref}`command-system`; 11 ordered counts total
  the complete draw count.

## 1. Frequency, recency, and draw structure

### 3. Statistics: Number Recency

- **Metadata:** **Proposed: existing capability**;
  `statistics.number-recency`; **Statistics**; keywords `recency`,
  `freshness`, `last seen`, `gap`, `overdue`.
- **Question/calculation:** When was each number last observed? Report its
  latest draw/date and zero-based current gap; identify never-observed numbers.
- **Parameters/result:** Number subset, default all 49; sortable table, gap
  bars, and explanation of zero- versus one-based gap conventions.
- **Consumers:** Freshness, Fresh Random, Bayesian, Markov 100, SVC, TBL,
  Lagged Logistic, score-grid and common ranking tie-breaks.
- **Behavior/acceptance:** Depends on command 1; every number appears once and
  latest-draw members have gap zero.

### 4. Statistics: Number Gap Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.number-gap-frequency`; **Statistics**; keywords `gap`,
  `recurrence`, `interval`, `waiting time`, `freshness`.
- **Question/calculation:** Which completed waiting intervals occur? Measure
  draw-index differences between consecutive appearances without treating the
  open current gap as a completed interval.
- **Parameters/result:** Number subset and optional maximum gap; per-number or
  pooled histogram and exact table.
- **Consumers:** Freshness, Bayesian, Markov 100, gap reports and recurrence
  features used by learned strategies.
- **Behavior/acceptance:** Depends on command 3; fewer than two appearances
  yield no completed interval, and direct occurrence-index walks match counts.

### 5. Statistics: Recent Number Frequency

- **Metadata:** **Proposed: new capability**;
  `statistics.recent-number-frequency`; **Statistics**; keywords `recent`,
  `window`, `frequency`, `rolling`, `rate`.
- **Question/calculation:** How does frequency change in a trailing window?
  Count each number over the last {math}`\min(w,N)` draws and divide by the
  effective window length.
- **Parameters/result:** Window defaults to 25; presets 5, 8, 10, 20, 24, 25
  and 100 plus validated custom input; bars/table with lifetime comparison.
- **Consumers:** Bayesian recency, SVC residuals, TBL, Border ML/SVC rolling
  features, Fresh Random and Predictive Score Grid.
- **Behavior/acceptance:** Depends on command 1; counts total six times the
  effective window and unavailable draws are not padded.

### 6. Statistics: Number Frequency Trend

- **Metadata:** **Proposed: existing capability**;
  `statistics.number-frequency-trend`; **Statistics**; keywords `trend`,
  `bins`, `rolling`, `number`, `time`.
- **Question/calculation:** Is a number's observed rate rising or falling over
  chronological bins? Report bin rates and a clearly descriptive slope.
- **Parameters/result:** Number subset and bin count, default current Statistics
  settings; multi-line chart, bin table and slope summary.
- **Consumers:** Numbers reports, TBL and Border Group ML/SVC trend features.
- **Behavior/acceptance:** Depends on commands 1 and 5; bins preserve order,
  cover every draw once and disclose unequal final-bin sizes.

### 7. Statistics: Position Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.position-frequency`; **Statistics**; keywords `position`,
  `sorted`, `frequency`, `slot`, `number`.
- **Question/calculation:** How often does a number occupy each sorted draw
  position? Count values 1--49 separately for positions 1--6.
- **Parameters/result:** Position defaults to all; heatmap and long-form
  count/rate table.
- **Consumers:** normalized-position Markov, position-based ML features and
  Numbers analysis.
- **Behavior/acceptance:** Each position totals {math}`N`; empty cells are
  zero rather than omitted.

### 8. Statistics: Draw Sum Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.draw-sum-frequency`; **Statistics**; keywords `sum`, `total`,
  `distribution`, `draw structure`.
- **Question/calculation:** How often does each six-number sum occur? Compute
  {math}`S_t=\sum_{i=1}^{6}x_{t,i}`.
- **Parameters/result:** Optional bin width, default exact integer sums;
  histogram/table with mean, median and central quantiles.
- **Consumers:** Overview, randomness diagnostics, draw similarity and learned
  structural features.
- **Behavior/acceptance:** Counts total {math}`N`; sums remain within valid
  bounds and binned totals equal exact totals.

### 9. Statistics: Parity Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.parity-frequency`; **Statistics**; keywords `odd`, `even`,
  `parity`, `composition`, `distribution`.
- **Question/calculation:** How many odd values occur per draw? Count zero
  through six; even count is {math}`6-k`.
- **Parameters/result:** None; seven bars and observed/expected composition
  table when the uniform null is available.
- **Consumers:** Overview, draw-shape scoring and SVC/ML structural features.
- **Behavior/acceptance:** Seven ordered rows, including zeros, total
  {math}`N`; each draw enters one row.

### 10. Statistics: Low/High Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.low-high-frequency`; **Statistics**; keywords `low`, `high`,
  `range`, `composition`, `24`.
- **Question/calculation:** How many values from 1--24 occur in each draw?
  Values 25--49 are high.
- **Parameters/result:** Production split 24/25; seven bars and exact table.
- **Consumers:** Overview, score-grid balance and learned structural features.
- **Behavior/acceptance:** Seven ordered rows total {math}`N`; 24 is low and
  25 is high.

### 11. Statistics: Consecutive Pair Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.consecutive-pair-frequency`; **Statistics**; keywords
  `consecutive`, `adjacent`, `pair`, `run`, `draw structure`.
- **Question/calculation:** How many adjacent sorted pairs differ by one in each
  draw? Count the five differences satisfying {math}`x_{i+1}-x_i=1`.
- **Parameters/result:** None; distribution bars and examples for each count.
- **Consumers:** Overview, nonlinear state vectors, Doublet/Triplet Markov and
  learned shape features.
- **Behavior/acceptance:** Counts range from zero to five and frequencies total
  {math}`N`.

### 12. Statistics: Draw Overlap Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.draw-overlap-frequency`; **Statistics**; keywords `overlap`,
  `matching`, `repeat`, `intersection`, `draw comparison`.
- **Question/calculation:** How many values are shared by two draws? Report
  {math}`|D_a\cap D_b|` and corresponding combination counts.
- **Parameters/result:** Consecutive draws by default; all pairs or a reference
  draw are alternatives; bars/table and selected-pair detail.
- **Consumers:** matching-combination randomness diagnostic, Draw Comparison,
  recurrence similarity and prediction audit.
- **Behavior/acceptance:** Requires two draws; overlap stays in 0--6 and the
  compared-pair count matches the selected mode.

## 2. Spaces, shape, and similarity

### 13. Statistics: Circular Space Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.circular-space-frequency`; **Statistics**; keywords `space`,
  `circular`, `gap`, `position`, `frequency`.
- **Question/calculation:** How often does each circular gap occur at each of
  the six sorted positions, including the wrap through 49?
- **Parameters/result:** Space position defaults to all six; heatmap and
  long-form counts/rates.
- **Consumers:** Markov Spaces, Border Group strategies, Space reports and
  Border ML/SVC current-space features.
- **Behavior/acceptance:** Six spaces per draw sum to 49; each position has
  {math}`N` observations.

### 14. Statistics: Circular Distance Frequency

- **Metadata:** **Proposed: existing capability**;
  `statistics.circular-distance-frequency`; **Statistics**; keywords
  `distance`, `spacing`, `aggregate`, `frequency`, `circle`.
- **Question/calculation:** How often does each circular space value occur after
  pooling all six positions?
- **Parameters/result:** Optional value range; bars/table with total and
  per-draw rate.
- **Consumers:** Spaces reports, Markov Spaces lifetime prior and shape models.
- **Behavior/acceptance:** Depends on command 13; pooled counts total
  {math}`6N` and equal the position-specific sum.

### 15. Statistics: Space Descriptive Summary

- **Metadata:** **Proposed: existing capability**;
  `statistics.space-descriptive`; **Statistics**; keywords `space`, `mean`,
  `median`, `standard deviation`, `quantile`.
- **Question/calculation:** What are the center and spread of each space
  position? Report count, mean, standard deviation, extremes and quartiles.
- **Parameters/result:** Position defaults to all; table and box plots.
- **Consumers:** Spaces reports, Markov Spaces analogue distances and learned
  shape features.
- **Behavior/acceptance:** Depends on command 13; values reproduce the six
  position columns and respect the row-wise sum of 49.

### 16. Statistics: Space Extremes

- **Metadata:** **Proposed: existing capability**;
  `statistics.space-extremes`; **Statistics**; keywords `minimum space`,
  `maximum space`, `extreme`, `distribution`, `range`.
- **Question/calculation:** How often does each per-draw minimum or maximum
  circular space occur?
- **Parameters/result:** Minimum, maximum or both; distributions and table.
- **Consumers:** Spaces reports, Border Group feature vectors and draw-shape
  diagnostics.
- **Behavior/acceptance:** Depends on command 13; each selected distribution
  totals {math}`N` and minimum never exceeds maximum.

### 17. Statistics: Normalized Position Frequency

- **Metadata:** **Proposed: new capability**;
  `statistics.normalized-position-frequency`; **Statistics**; keywords
  `normalized`, `position`, `deviation`, `slot`, `distribution`.
- **Question/calculation:** How are sorted positions distributed under the exact
  position transformation used by production?
- **Parameters/result:** Position and bin count; faceted histograms/table with
  the production formula printed in the explanation.
- **Consumers:** Markov Normalized Positions and learned position features.
- **Behavior/acceptance:** Depends on command 7; raw and transformed values are
  traceable and the transform matches production.

### 18. Statistics: Relative Dispersion Frequency

- **Metadata:** **Proposed: new capability**;
  `statistics.relative-dispersion-frequency`; **Statistics**; keywords
  `relative dispersion`, `spread`, `range`, `normalized`, `distribution`.
- **Question/calculation:** How often does each production relative-dispersion
  state occur?
- **Parameters/result:** Bin/state selection; histogram and state table with
  the defining formula.
- **Consumers:** Markov Relative Dispersion, recurrence and shape strategies.
- **Behavior/acceptance:** Depends on commands 8 and 15; state construction
  matches production and counts total {math}`N`.

### 19. Statistics: Draw Similarity

- **Metadata:** **Proposed: new capability**; `statistics.draw-similarity`;
  **Statistics**; keywords `similarity`, `distance`, `nearest draw`,
  `analogue`, `shape`.
- **Question/calculation:** Which historical draws are closest to a selected
  reference under an available production distance?
- **Parameters/result:** Reference draw, distance metric and result count;
  ranked table, distance bars and draw details.
- **Consumers:** Recurrence Dynamics, Markov Spaces analogues, Predictive Score
  Grid and Draw Comparison.
- **Behavior/acceptance:** Requires two draws; excludes the reference, uses
  deterministic ties and reproduces direct metric calculations.

### 20. Statistics: EMD Successor Similarity

- **Metadata:** **Proposed: new capability**;
  `statistics.emd-successor-similarity`; **Statistics**; keywords
  `earth mover`, `wasserstein`, `successor`, `similarity`, `emd`.
- **Question/calculation:** Which predecessor draws are closest to the reference
  under one-dimensional EMD, and what occurred in their successors?
- **Parameters/result:** Reference draw, neighbour count and optional distance
  weighting; neighbour/successor table and candidate score bars.
- **Consumers:** Earth Mover Distance and Predictive Score Grid.
- **Behavior/acceptance:** Depends on command 19; only predecessors with a known
  successor contribute, replay is future-safe, and distances match production.

## 3. Relationships, inference, and randomness

### 21. Statistics: Pair Co-occurrence

- **Metadata:** **Proposed: existing capability**;
  `statistics.pair-cooccurrence`; **Statistics**; keywords `pair`,
  `co-occurrence`, `relationship`, `count`, `matrix`.
- **Question/calculation:** How many draws contain each unordered number pair?
- **Parameters/result:** Optional anchor number or all pairs; symmetric heatmap
  and pair table.
- **Consumers:** Co-occurrence, Predictive Score Grid, TBL, correlations,
  portfolio diversity and Relationships reports.
- **Behavior/acceptance:** Every draw contributes 15 unordered pairs; the upper
  triangle totals {math}`15N`.

### 22. Statistics: Co-occurrence Lift

- **Metadata:** **Proposed: existing capability**;
  `statistics.cooccurrence-lift`; **Statistics**; keywords `lift`,
  `expected`, `pair`, `association`, `residual`.
- **Question/calculation:** Is a pair above or below its independence
  expectation? Report observed, expected, lift, deviation and marginals.
- **Parameters/result:** Anchor/all pairs and minimum support; diverging heatmap
  plus auditable table.
- **Consumers:** Co-occurrence, Relationships, Predictive Score Grid and model
  compatibility features.
- **Behavior/acceptance:** Depends on commands 1 and 21; zero expectations are
  guarded, symmetry holds and every lift is reconstructible.

### 23. Statistics: Doublet Frequency

- **Metadata:** **Proposed: new capability**;
  `statistics.doublet-frequency`; **Statistics**; keywords `doublet`,
  `consecutive`, `run`, `start`, `frequency`.
- **Question/calculation:** How often does each consecutive two-number run begin
  in historical draws?
- **Parameters/result:** Optional start range; bars/table with draw dates.
- **Consumers:** Doublet/Triplet Markov and consecutive-shape diagnostics.
- **Behavior/acceptance:** Depends on command 11; overlapping runs follow
  production and direct draw scans reproduce counts.

### 24. Statistics: Triplet Frequency

- **Metadata:** **Proposed: new capability**;
  `statistics.triplet-frequency`; **Statistics**; keywords `triplet`,
  `consecutive`, `run`, `start`, `frequency`.
- **Question/calculation:** How often does each consecutive three-number run
  begin in historical draws?
- **Parameters/result:** Optional start range; bars/table with draw dates.
- **Consumers:** Doublet/Triplet Markov and consecutive-shape diagnostics.
- **Behavior/acceptance:** Depends on commands 11 and 23; overlapping runs
  follow production and direct draw scans reproduce counts.

### 25. Statistics: Pearson Correlation

- **Metadata:** **Proposed: existing capability**;
  `statistics.pearson-correlation`; **Statistics**; keywords `pearson`,
  `correlation`, `linear`, `matrix`, `relationship`.
- **Question/calculation:** What linear association is present among number
  indicators or among space positions?
- **Parameters/result:** Domain defaults to numbers, with spaces optional;
  heatmap and selected-cell details.
- **Consumers:** Relationships, learned models and feature-dependence checks.
- **Behavior/acceptance:** Constant columns are explained as undefined; the
  matrix is symmetric with unit diagonal where defined.

### 26. Statistics: Spearman Correlation

- **Metadata:** **Proposed: existing capability**;
  `statistics.spearman-correlation`; **Statistics**; keywords `spearman`,
  `rank`, `correlation`, `monotonic`, `matrix`.
- **Question/calculation:** What monotonic rank association is present among
  the selected variables?
- **Parameters/result:** Number or space domain; heatmap, selected-cell ranks
  and coefficient details.
- **Consumers:** Relationships and model feature diagnostics.
- **Behavior/acceptance:** Uses tied ranks consistently, explains constant
  variables and agrees with a reference calculation.

### 27. Statistics: Number/Space Correlation

- **Metadata:** **Proposed: existing capability**;
  `statistics.number-space-correlation`; **Statistics**; keywords `number`,
  `space`, `cross correlation`, `position`, `relationship`.
- **Question/calculation:** How is each number indicator associated with each
  circular-space position?
- **Parameters/result:** Pearson or Spearman, default current setting; 49-by-6
  heatmap and selected-cell detail.
- **Consumers:** Relationships and Spaces, Markov Spaces and learned features.
- **Behavior/acceptance:** Depends on commands 13, 25 and 26; dimensions and
  method are explicit and coefficients reproduce source data.

### 28. Statistics: Frequency Chi-square

- **Metadata:** **Proposed: existing capability**;
  `statistics.frequency-chi-square`; **Statistics**; keywords `chi-square`,
  `uniform`, `frequency`, `p-value`, `residual`.
- **Question/calculation:** How far is the 49-number frequency vector from its
  uniform marginal expectation? Report statistic, degrees of freedom, p-value
  and per-number contributions.
- **Parameters/result:** None; contribution bars/table and test summary.
- **Consumers:** Randomness and Chi-square Frequency interpretation.
- **Behavior/acceptance:** Depends on command 1; contributions sum to the global
  statistic and the result distinguishes significance from strategy ranking.

### 29. Statistics: Categorical Chi-square

- **Metadata:** **Proposed: existing capability**;
  `statistics.categorical-chi-square`; **Statistics**; keywords
  `categorical`, `chi-square`, `contingency`, `cramers-v`, `residual`.
- **Question/calculation:** Is a candidate hit associated with its selected
  categorical state? Report contingency counts, chi-square, p-value, corrected
  Cramer's V and cell residuals.
- **Parameters/result:** Candidate number and production state dimensions;
  contingency/residual heatmaps and test table.
- **Consumers:** Categorical Chi-square and Prediction Audit.
- **Behavior/acceptance:** Sparse expected cells are explained; totals and
  statistics match the causal production table.

### 30. Statistics: Frequency Entropy

- **Metadata:** **Proposed: existing capability**;
  `statistics.frequency-entropy`; **Statistics**; keywords `entropy`,
  `shannon`, `uniformity`, `frequency`, `information`.
- **Question/calculation:** How evenly are appearances distributed? Compute
  Shannon entropy normalized against the 49-number maximum.
- **Parameters/result:** Whole history or validated trailing window;
  contribution bars and entropy summary.
- **Consumers:** Entropy and Randomness.
- **Behavior/acceptance:** Depends on command 1 or 5; zero probabilities
  contribute zero, normalized entropy stays in 0--1 and uniform reaches one.

### 31. Statistics: Randomness Summary

- **Metadata:** **Dependent**; `statistics.randomness-summary`;
  **Statistics**; keywords `randomness`, `diagnostic`, `entropy`, `overlap`.
- **Question/calculation:** Combine frequency chi-square, normalized entropy,
  draw-sum lag-one autocorrelation and matching-combination diagnostics.
- **Parameters/result:** Lag defaults to one; educational cards and component
  tables without one composite verdict.
- **Consumers:** Randomness, Autocorrelation and strategy audit.
- **Behavior/acceptance:** Depends on commands 8, 12, 28 and 30; every card
  reproduces its standalone command and explains multiplicity.

## 4. Markov, sequence, and recurrence

### 32. Statistics: Last-draw Transition

- **Metadata:** **Proposed: new capability**;
  `statistics.last-draw-transition`; **Statistics**; keywords `transition`,
  `previous draw`, `successor`, `conditional`, `rate`.
- **Question/calculation:** Given a number in the preceding draw, how often did
  each candidate occur in the next draw? Report transition counts, opportunities
  and smoothed rates without calling them a joint ticket probability.
- **Parameters/result:** Source number or latest six; directed heatmap/table.
- **Consumers:** Predictive Score Grid, Co-occurrence and sequence diagnostics.
- **Behavior/acceptance:** Requires two draws; every transition uses only a
  predecessor and its known successor and matches a chronological scan.

### 33. Statistics: Gap-state Transition

- **Metadata:** **Proposed: new capability**;
  `statistics.gap-state-transition`; **Statistics**; keywords `gap state`,
  `markov`, `transition`, `bucket`, `hit rate`.
- **Question/calculation:** What historical hit rate followed each pre-draw gap
  bucket? Report opportunities, hits, smoothing and optional decay.
- **Parameters/result:** Number, bucket cap and production model preset;
  probability curve/table with effective sample sizes.
- **Consumers:** Markov 100, Bayesian gap components and Predictive Score Grid.
- **Behavior/acceptance:** Depends on commands 3 and 4; gaps are captured before
  outcomes, the capped tail is explicit and future-draw changes cannot alter an
  earlier row.

### 34. Statistics: Freshness Markov

- **Metadata:** **Proposed: new capability**;
  `statistics.freshness-markov`; **Statistics**; keywords `freshness`, `binary`,
  `variable order`, `markov`, `context`.
- **Question/calculation:** What hit/failure counts follow a number's binary
  appearance suffix at each supported context order?
- **Parameters/result:** Number and order 1--20; context table, probability and
  lift against base rate.
- **Consumers:** Markov Freshness.
- **Behavior/acceptance:** Depends on command 33; training precedes appending
  the current outcome, sparse contexts show fallback, and replay is prefix-safe.

### 35. Statistics: Space Markov

- **Metadata:** **Proposed: new capability**; `statistics.space-markov`;
  **Statistics**; keywords `space`, `markov`, `context`, `analogue`, `successor`.
- **Question/calculation:** Which exact or similar circular-space contexts
  preceded each next space value?
- **Parameters/result:** Position, order 1--20 and exact/analogue branch;
  transition table, similarity weights and successor distribution.
- **Consumers:** Markov Spaces and Predictive Score Grid.
- **Behavior/acceptance:** Depends on commands 13--15 and 19; context updates are
  walk-forward, position totals reconcile and weights match production.

### 36. Statistics: Normalized-position Markov

- **Metadata:** **Proposed: new capability**;
  `statistics.normalized-position-markov`; **Statistics**; keywords
  `normalized position`, `markov`, `state`, `transition`, `context`.
- **Question/calculation:** What next normalized-position states followed the
  selected production state/context?
- **Parameters/result:** Position, state and context order; transition matrix,
  row counts and probabilities.
- **Consumers:** Markov Normalized Positions.
- **Behavior/acceptance:** Depends on command 17; states and fallbacks reproduce
  production and each nonempty probability row sums to one.

### 37. Statistics: Relative-dispersion Markov

- **Metadata:** **Proposed: new capability**;
  `statistics.relative-dispersion-markov`; **Statistics**; keywords
  `relative dispersion`, `markov`, `state`, `transition`, `spread`.
- **Question/calculation:** What next dispersion states followed each production
  relative-dispersion context?
- **Parameters/result:** State and context order; transition matrix, counts and
  probabilities.
- **Consumers:** Markov Relative Dispersion.
- **Behavior/acceptance:** Depends on command 18; state generation matches
  production, empty rows explain fallback and nonempty rows sum to one.

### 38. Statistics: Doublet/Triplet Markov

- **Metadata:** **Proposed: new capability**;
  `statistics.doublet-triplet-markov`; **Statistics**; keywords `doublet`,
  `triplet`, `markov`, `run`, `transition`.
- **Question/calculation:** Which consecutive run starts followed earlier
  doublet or triplet states?
- **Parameters/result:** Run size and source start; directed transition table
  and candidate support bars.
- **Consumers:** Doublet/Triplet Markov.
- **Behavior/acceptance:** Depends on commands 23 and 24; opportunity counts,
  overlap handling and smoothing reproduce production causally.

### 39. Statistics: Recurrence Plot

- **Metadata:** **Proposed: existing capability**;
  `statistics.recurrence-plot`; **Statistics**; keywords `recurrence`, `plot`,
  `distance`, `threshold`, `nonlinear`.
- **Question/calculation:** Which draw-state pairs are recurrent under the
  selected distance and threshold?
- **Parameters/result:** State representation, distance, threshold and optional
  trailing length; recurrence-matrix heatmap and recurrence rate.
- **Consumers:** Recurrence Dynamics and Nonlinear Dynamics.
- **Behavior/acceptance:** Depends on command 19; matrix symmetry/diagonal rules
  are explicit and a direct thresholded distance matrix matches it.

### 40. Statistics: Recurrence Quantification

- **Metadata:** **Proposed: existing capability**;
  `statistics.recurrence-quantification`; **Statistics**; keywords `rqa`,
  `recurrence rate`, `determinism`, `laminarity`, `entropy`.
- **Question/calculation:** What recurrence-quantification measures describe the
  selected recurrence matrix?
- **Parameters/result:** Reuses command 39 parameters plus minimum line lengths;
  metric cards, line-length distributions and definitions.
- **Consumers:** Recurrence Dynamics and Nonlinear Dynamics.
- **Behavior/acceptance:** Depends on command 39; undefined short-history
  metrics are explained and line counts reconstruct each reported measure.

### 41. Statistics: Recurrence Successors

- **Metadata:** **Dependent**; `statistics.recurrence-successors`;
  **Statistics**; keywords `recurrence`, `nearest neighbour`, `successor`,
  `forecast evidence`, `analogue`.
- **Question/calculation:** What happened after the nearest recurrent historical
  states, and how does that evidence distribute over numbers?
- **Parameters/result:** Reference draw, neighbour count, distance and weighting;
  neighbour/successor table and candidate evidence bars.
- **Consumers:** Recurrence Dynamics, recurrence hybrids and Prediction Audit.
- **Behavior/acceptance:** Depends on commands 19, 39 and 40; neighbours require
  known successors, exclude future information and use deterministic ties.

## 5. Border Space Groups

### 42. Statistics: Border Null Signatures

- **Metadata:** **Proposed: existing capability**;
  `statistics.border-null-signatures`; **Statistics**; keywords `border`,
  `signature`, `null`, `expected`, `exact probability`.
- **Question/calculation:** What exact signature and group-count probabilities
  follow from uniform six-number combinations at a selected Border space?
- **Parameters/result:** Border space defaults to Settings; expected-probability
  bars/table for all 11 signatures and six group counts.
- **Consumers:** Border Group Statistical, Bayesian, ML/SVC baselines and Space
  Groups inference.
- **Behavior/acceptance:** Depends on command 2; probabilities sum to one,
  aggregate by group count and match exact combinatorial enumeration.

### 43. Statistics: Border Signature Residuals

- **Metadata:** **Proposed: existing capability**;
  `statistics.border-signature-residuals`; **Statistics**; keywords `border`,
  `signature`, `residual`, `chi-square`, `expected`.
- **Question/calculation:** Which signature counts differ most from the exact
  null? Report observed/expected counts, Pearson residuals, contributions,
  global chi-square and p-value.
- **Parameters/result:** Border space; diverging residual bars and table.
- **Consumers:** Border Group Statistical and Space Groups.
- **Behavior/acceptance:** Depends on commands 2 and 42; contributions sum to
  the global statistic and zero-count signatures remain present.

### 44. Statistics: Border Transition Matrix

- **Metadata:** **Proposed: existing capability**;
  `statistics.border-transition-matrix`; **Statistics**; keywords `border`,
  `signature`, `transition`, `markov`, `matrix`.
- **Question/calculation:** How often did each group signature follow every
  previous signature?
- **Parameters/result:** Border space and count/row-percentage view; 11-by-11
  heatmap and table.
- **Consumers:** Border Group Markov, Bayesian and Hybrid.
- **Behavior/acceptance:** Depends on command 2 and two draws; transition totals
  equal {math}`N-1` and nonempty row percentages sum to 100%.

### 45. Statistics: Border Transition Lift

- **Metadata:** **Proposed: existing capability**;
  `statistics.border-transition-lift`; **Statistics**; keywords `border`,
  `transition`, `lift`, `expected`, `residual`.
- **Question/calculation:** Which signature transitions occur above or below
  their marginal-independence expectations?
- **Parameters/result:** Border space and minimum support; expected-count and
  lift heatmaps with auditable table.
- **Consumers:** Border Group Markov and Space Groups.
- **Behavior/acceptance:** Depends on commands 42 and 44; zero expectations are
  guarded and observed/expected/lift values reconcile cell by cell.

### 46. Statistics: Border Transition Information

- **Metadata:** **Proposed: existing capability**;
  `statistics.border-transition-information`; **Statistics**; keywords
  `border`, `mutual information`, `permutation`, `transition`, `p-value`.
- **Question/calculation:** How much information about the next signature is in
  the previous signature, and how unusual is it under sequence permutations?
- **Parameters/result:** Border space and validated permutation count; observed
  mutual information, null histogram and permutation p-value.
- **Consumers:** Border Group Markov interpretation and Space Groups.
- **Behavior/acceptance:** Depends on command 44; deterministic seeded
  permutations preserve marginals and p-value correction is documented.

### 47. Statistics: Border Threshold Sensitivity

- **Metadata:** **Proposed: existing capability**;
  `statistics.border-threshold-sensitivity`; **Statistics**; keywords `border`,
  `threshold`, `sensitivity`, `groups`, `signatures`.
- **Question/calculation:** How do group-count and signature distributions
  change across feasible Border spaces?
- **Parameters/result:** Validated border range, default app sensitivity range;
  multi-line chart and border-by-signature table.
- **Consumers:** every Border Group strategy and Space Groups.
- **Behavior/acceptance:** Depends on command 2; each border classifies every
  draw once and the currently selected border is highlighted.

### 48. Statistics: Border Window Features

- **Metadata:** **Proposed: new capability**;
  `statistics.border-window-features`; **Statistics**; keywords `border`,
  `window`, `10`, `25`, `100`, `frequency`, `average groups`.
- **Question/calculation:** What signature frequencies and average group counts
  are available over the production 10-, 25- and 100-draw windows?
- **Parameters/result:** Border space and reference draw; comparison table and
  grouped window bars.
- **Consumers:** Border Group ML and SVC.
- **Behavior/acceptance:** Depends on commands 2 and 5; windows truncate rather
  than pad and a replay reference never sees its outcome or future draws.

### 49. Statistics: Border Feature Vector

- **Metadata:** **Dependent**; `statistics.border-feature-vector`;
  **Strategy Diagnostics**; keywords `border`, `feature vector`, `ml`, `svc`,
  `leakage safe`.
- **Question/calculation:** What exact production inputs represent one pre-draw
  Border ML/SVC example: three signatures, group counts and maxima, current
  spaces, rolling summaries and trend?
- **Parameters/result:** Border space, reference draw and ML/SVC model; indexed
  feature table grouped by source with value provenance.
- **Consumers:** Border Group ML and SVC.
- **Behavior/acceptance:** Depends on commands 13, 16, 47 and 48; vector length,
  order and values match production and later draws cannot change it.

### 50. Statistics: Border Class Balance

- **Metadata:** **Proposed: new capability**;
  `statistics.border-class-balance`; **Strategy Diagnostics**; keywords
  `border`, `class balance`, `signature`, `training`, `svc`.
- **Question/calculation:** Which of the 11 signature classes are represented in
  the selected causal training window and with what balanced-class weights?
- **Parameters/result:** Border space, reference draw, model and rolling limit;
  class bars/table with feasible, observed and weighted status.
- **Consumers:** Border Group ML/SVC training and Hybrid interpretation.
- **Behavior/acceptance:** Depends on command 49; label total equals the actual
  training examples, unseen classes remain visible and one-class histories
  explain baseline fallback.

## 6. Model and strategy diagnostics

These commands run a causal whole-database replay when their result depends on
training or forecast history. A **Strategy** quick-pick lists only compatible
models and explains why an unavailable model is disabled.

### 51. Strategy: Score Breakdown

- **Metadata:** **Proposed: new capability**; `strategy.score-breakdown`;
  **Strategy Diagnostics**; keywords `score`, `explain`, `component`, `rank`.
- **Question/calculation:** Which raw, transformed and weighted components form
  a selected candidate's final production score?
- **Parameters/result:** Strategy, reference draw and candidate; calculation
  waterfall, component table and tie-break explanation.
- **Consumers:** every non-random ranking strategy and Prediction Audit.
- **Behavior/acceptance:** Component arithmetic reconstructs the serialized
  score within tolerance; hidden future state is unavailable.

### 52. Strategy: Ranking Distribution

- **Metadata:** **Proposed: new capability**;
  `strategy.ranking-distribution`; **Strategy Diagnostics**; keywords `rank`,
  `score distribution`, `top 6`, `candidate`, `spread`.
- **Question/calculation:** How are all 49 scores and ranks distributed for one
  forecast, and where is the Top-6 boundary?
- **Parameters/result:** Strategy and reference draw; ranked bars/table with
  score range, ties, current gaps and Top-6 membership.
- **Consumers:** all ranking strategies, Predictions and Prediction Audit.
- **Behavior/acceptance:** Depends on command 51; all 49 candidates occur once
  and displayed sorting reproduces production ranking and tie rules.

### 53. Strategy: Feature Vector

- **Metadata:** **Proposed: new capability**; `strategy.feature-vector`;
  **Strategy Diagnostics**; keywords `feature`, `vector`, `input`, `ml`, `scale`.
- **Question/calculation:** What exact raw and transformed feature values does a
  learned strategy receive for one candidate/example?
- **Parameters/result:** Compatible strategy, reference draw and candidate;
  indexed table with formula, raw value, transformed value and source window.
- **Consumers:** SVC, TBL, Scikit Online SVM, Lagged Logistic, sparse-neural,
  Border ML/SVC and Decision Tree Selector.
- **Behavior/acceptance:** Feature count/order matches production and changing a
  later draw cannot alter the selected vector.

### 54. Strategy: Training Window

- **Metadata:** **Proposed: new capability**; `strategy.training-window`;
  **Strategy Diagnostics**; keywords `training`, `window`, `warm-up`, `rolling`,
  `retrain`.
- **Question/calculation:** Which labeled examples train the selected model at a
  reference draw, and why is it fitted or using fallback?
- **Parameters/result:** Learned strategy and reference draw; timeline/table of
  warm-up, retained examples, class count and retraining boundaries.
- **Consumers:** every learned strategy, especially Border SVC's 50-example
  warm-up, 500-example limit and 25-example retraining schedule.
- **Behavior/acceptance:** Depends on command 53; no label predates its feature
  snapshot, limits/schedules match production and fallback reason is explicit.

### 55. Strategy: Markov Context

- **Metadata:** **Dependent**; `strategy.markov-context`;
  **Strategy Diagnostics**; keywords `markov`, `context`, `order`, `fallback`,
  `transition`.
- **Question/calculation:** Which current context rows, counts and fallback
  levels contribute to a selected Markov forecast?
- **Parameters/result:** Markov strategy, reference draw and candidate/state;
  context ladder with raw counts, smoothing, confidence and contribution.
- **Consumers:** Markov 100, MKGSV, Freshness, Spaces, Normalized Positions,
  Relative Dispersion, Doublet/Triplet and Border Markov.
- **Behavior/acceptance:** Depends on commands 32--38 and 44; selected context
  and fallback reproduce production without future observations.

### 56. Strategy: Bayesian Posterior

- **Metadata:** **Proposed: new capability**; `strategy.bayesian-posterior`;
  **Strategy Diagnostics**; keywords `bayesian`, `posterior`, `prior`,
  `shrinkage`, `half-life`.
- **Question/calculation:** How do prior strength, observations, hits and decay
  produce each posterior-mean component and final model average?
- **Parameters/result:** Bayesian or Border Bayesian, reference draw and
  candidate/signature; prior/evidence/posterior table and blend waterfall.
- **Consumers:** Bayesian and Border Group Bayesian/Hybrid.
- **Behavior/acceptance:** Depends on commands 33 and 44; effective counts and
  posterior formulas reconstruct production and sparse evidence stays finite.

### 57. Strategy: Probability Distribution

- **Metadata:** **Proposed: existing capability**;
  `strategy.probability-distribution`; **Strategy Diagnostics**; keywords
  `probability`, `class`, `candidate`, `forecast`, `normalization`.
- **Question/calculation:** What complete probability/score mass does a selected
  strategy assign before final ticket decoding?
- **Parameters/result:** Strategy and reference draw; 49 candidates or 11
  signatures as bars/table, including zero and infeasible classes.
- **Consumers:** probabilistic, random, ML and Border Group strategies.
- **Behavior/acceptance:** Values map to the complete domain, normalized models
  sum to one and target-group conditioning is shown separately.

### 58. Strategy: Support Vectors

- **Metadata:** **Proposed: new capability**; `strategy.support-vectors`;
  **Strategy Diagnostics**; keywords `svc`, `svm`, `support vector`, `margin`,
  `kernel`.
- **Question/calculation:** How many support vectors represent each observed
  class and what are the fitted SVC/scaler settings?
- **Parameters/result:** SVC-compatible strategy and fitted reference draw;
  class counts, model configuration and summarized scaled-vector diagnostics.
- **Consumers:** Support Vector Classifier, Scikit Online SVM where applicable,
  and Border Group SVC.
- **Behavior/acceptance:** Depends on command 54; unavailable before fitting,
  counts match the fitted estimator and sensitive raw training rows are not
  duplicated unnecessarily.

### 59. Strategy: Online Model Weights

- **Metadata:** **Proposed: new capability**;
  `strategy.online-model-weights`; **Strategy Diagnostics**; keywords `online`,
  `weights`, `coefficient`, `learning`, `residual`.
- **Question/calculation:** What coefficients or learned residual weights does
  an online model hold at a selected replay point?
- **Parameters/result:** Compatible model and reference draw; signed weight
  bars/table with feature scaling and update history summary.
- **Consumers:** TBL, Lagged Logistic, Border Group ML and related online models.
- **Behavior/acceptance:** Depends on commands 53 and 54; weights match the
  prefix-fitted state and displayed contributions preserve sign and scale.

### 60. Strategy: Tree Feature Importance

- **Metadata:** **Proposed: new capability**;
  `strategy.tree-feature-importance`; **Strategy Diagnostics**; keywords `tree`,
  `feature importance`, `selector`, `split`, `decision`.
- **Question/calculation:** Which features and split decisions drive the fitted
  Decision Tree Selector at the selected replay point?
- **Parameters/result:** Reference draw and optional tree depth; importance bars
  and navigable split table.
- **Consumers:** Decision Tree Selector.
- **Behavior/acceptance:** Depends on commands 53 and 54; importances match the
  fitted tree, sum as defined by the estimator and do not imply causality.

### 61. Strategy: Calibration

- **Metadata:** **Proposed: new capability**; `strategy.calibration`;
  **Strategy Diagnostics**; keywords `calibration`, `reliability`, `probability`,
  `forecast`, `bin`.
- **Question/calculation:** Do forecast probabilities correspond to observed
  event rates across causal evaluation bins?
- **Parameters/result:** Probabilistic strategy, event definition and bins;
  reliability plot/table with support and optional calibration error.
- **Consumers:** Bayesian, SVC, Markov and Border probabilistic strategies.
- **Behavior/acceptance:** Depends on command 57 and at least two populated
  bins; bin totals equal evaluated events and no training outcomes are scored
  as if they were unseen forecasts.

### 62. Strategy: Walk-forward Log Loss

- **Metadata:** **Proposed: existing capability**;
  `strategy.walk-forward-log-loss`; **Strategy Diagnostics**; keywords
  `log loss`, `cross entropy`, `walk forward`, `model metric`.
- **Question/calculation:** What mean negative log probability did the model
  assign to actual outcomes over causal forecasts?
- **Parameters/result:** Strategy, trailing evaluation window default 100, and
  optional date range; loss timeline, summary and forecast table.
- **Consumers:** Border model metrics and adaptive Hybrid weights; all
  probabilistic model comparisons.
- **Behavior/acceptance:** Depends on command 57; probabilities are safely
  bounded, evaluated forecasts are counted once and mean matches row losses.

### 63. Strategy: Walk-forward Brier Score

- **Metadata:** **Proposed: existing capability**;
  `strategy.walk-forward-brier`; **Strategy Diagnostics**; keywords `brier`,
  `quadratic`, `walk forward`, `probability`, `model metric`.
- **Question/calculation:** What mean squared probability error did the model
  produce over causal forecasts?
- **Parameters/result:** Strategy, event encoding and trailing/date range;
  score timeline, summary and forecast table.
- **Consumers:** Border model metrics and probabilistic model comparison.
- **Behavior/acceptance:** Depends on command 57; one-hot encoding and class
  domain are explicit and the mean reconstructs from per-forecast scores.

### 64. Strategy: Accuracy

- **Metadata:** **Proposed: existing capability**; `strategy.accuracy`;
  **Strategy Diagnostics**; keywords `accuracy`, `signature`, `group count`,
  `top 6`, `mae`.
- **Question/calculation:** How often did the selected point prediction match
  its actual class or Top-6 event, and what group-count MAE applies?
- **Parameters/result:** Strategy, accuracy definition and range; accuracy/MAE
  cards and outcome table.
- **Consumers:** Border model metrics, Predictions and Strategy Effectiveness.
- **Behavior/acceptance:** Depends on command 57; numerator, denominator and
  warm-up exclusions are shown and reproduce the displayed metric.

### 65. Strategy: Confusion Matrix

- **Metadata:** **Proposed: new capability**; `strategy.confusion-matrix`;
  **Strategy Diagnostics**; keywords `confusion`, `classification`, `actual`,
  `predicted`, `class`.
- **Question/calculation:** Which actual classes are confused with which
  predicted classes in walk-forward forecasts?
- **Parameters/result:** Classifier, signature/group-count target and range;
  count and normalized heatmaps with support table.
- **Consumers:** Border ML/SVC, Decision Tree Selector and classifiers.
- **Behavior/acceptance:** Depends on command 64; all domain classes remain
  visible, cell total equals evaluated forecasts and normalization is labeled.

### 66. Strategy: Ensemble Weights

- **Metadata:** **Proposed: existing capability**;
  `strategy.ensemble-weights`; **Strategy Diagnostics**; keywords `ensemble`,
  `hybrid`, `weight`, `adaptive`, `component`.
- **Question/calculation:** What component weights does an ensemble use now and
  how were adaptive weights derived?
- **Parameters/result:** Ensemble and reference draw; weight bars, history and
  source-metric table.
- **Consumers:** Mixed, SVC recurrence hybrids, SRPH variants, Border Hybrid,
  CIS, Residual Coverage and Chained.
- **Behavior/acceptance:** Depends on command 62 where adaptive; weights sum to
  one, minimum floors are visible and the blend reproduces production.

### 67. Strategy: Component Contributions

- **Metadata:** **Dependent**; `strategy.component-contributions`;
  **Strategy Diagnostics**; keywords `ensemble`, `contribution`, `component`,
  `score`, `waterfall`.
- **Question/calculation:** How much does each weighted component contribute to
  one candidate's final ensemble score?
- **Parameters/result:** Ensemble, reference draw and candidate; contribution
  waterfall and aligned component rank/score table.
- **Consumers:** all ensembles and Predictive Score Grid.
- **Behavior/acceptance:** Depends on commands 51 and 66; contribution sum
  reconstructs the final score and normalization stages are separated.

### 68. Strategy: Agreement

- **Metadata:** **Proposed: new capability**; `strategy.agreement`;
  **Strategy Diagnostics**; keywords `agreement`, `rank correlation`, `overlap`,
  `strategy comparison`, `diversity`.
- **Question/calculation:** How similar are strategy rankings and Top-6 sets over
  the selected walk-forward period?
- **Parameters/result:** Strategy set, rank/Top-6 metric and range; pairwise
  heatmap and distribution table.
- **Consumers:** ensemble selection, SRPH and Strategy Effectiveness.
- **Behavior/acceptance:** Depends on command 52; matrices are symmetric,
  diagonals are exact and each comparison uses aligned forecast dates.

### 69. Strategy: Residual Diversity

- **Metadata:** **Proposed: new capability**;
  `strategy.residual-diversity`; **Strategy Diagnostics**; keywords `residual`,
  `diversity`, `error`, `ensemble`, `srph`.
- **Question/calculation:** How different are component forecast residuals after
  accounting for their common signal?
- **Parameters/result:** Compatible ensemble/component set and range; residual
  correlation heatmap, diversity scores and selected-pair timeline.
- **Consumers:** SRPH Residual Diversity Hybrid and ensemble diagnostics.
- **Behavior/acceptance:** Depends on commands 61--64 and 68; residual definition
  is explicit, dates align and production diversity scores are reproduced.

### 70. Strategy: Minimax Regret

- **Metadata:** **Proposed: new capability**; `strategy.minimax-regret`;
  **Strategy Diagnostics**; keywords `minimax`, `regret`, `ensemble`, `worst
  case`, `srph`.
- **Question/calculation:** What regret does each candidate/weighting incur
  relative to the best component under each evaluated scenario?
- **Parameters/result:** Compatible ensemble and range; regret matrix, worst-case
  bars and chosen minimax solution.
- **Consumers:** SRPH Minimax Regret Hybrid.
- **Behavior/acceptance:** Depends on commands 66--68; scenario definitions are
  visible and the selected solution has the minimum displayed maximum regret.

### 71. Strategy: Coverage

- **Metadata:** **Proposed: existing capability**; `strategy.coverage`;
  **Strategy Diagnostics**; keywords `coverage`, `portfolio`, `diversity`,
  `residual`, `ticket`.
- **Question/calculation:** How broadly do selected tickets cover candidates,
  pairs, strategy evidence or residual opportunity?
- **Parameters/result:** Coverage definition, strategy/ticket set and portfolio
  size; coverage matrix, uncovered items and marginal-gain sequence.
- **Consumers:** Residual Coverage, portfolio generation and backtesting.
- **Behavior/acceptance:** Depends on commands 21, 52 and 69 as selected; every
  covered item traces to a ticket and reported totals match set operations.

### 72. Strategy: Effectiveness

- **Metadata:** **Proposed: existing capability**; `strategy.effectiveness`;
  **Strategy Diagnostics**; keywords `effectiveness`, `backtest`, `hits`,
  `history`, `strategy`.
- **Question/calculation:** How many next-draw hits did each completed causal
  forecast obtain, with what mean and distribution?
- **Parameters/result:** Strategies, date/trailing range and optional family;
  hit timeline, distribution, summary table and sample sizes.
- **Consumers:** Strategy Effectiveness, ensemble evaluation and portfolio
  backtests.
- **Behavior/acceptance:** Forecasts are generated before their outcomes,
  incomplete final forecasts are excluded and summary totals match history.

### 73. Strategy: Prediction Audit

- **Metadata:** **Dependent**; `strategy.prediction-audit`;
  **Strategy Diagnostics**; keywords `prediction`, `audit`, `provenance`,
  `leakage`, `explain`.
- **Question/calculation:** Can one forecast be traced from its data prefix and
  parameters through features, model state, scores, ranking and later outcome?
- **Parameters/result:** Strategy and forecast date; educational audit timeline
  linking the relevant command results and stored prediction details.
- **Consumers:** Prediction Audit, model validation and contributor debugging.
- **Behavior/acceptance:** Depends on commands 51--72 as applicable; provenance
  identifies every input boundary, scores/ranks reconcile and unavailable
  future outcome is clearly distinguished from calculation failure.

## General application commands

**Existing** exposes an action already present; **New** needs supporting
behavior. **Read** has no persistent effect, **Write** changes recoverable
state, and **Destructive** can remove or overwrite user-managed state. Write
and destructive commands preview the exact target/effect before confirmation.
Cancellation changes nothing and restores focus.

### Datasets and reports

| Command | Status/effect | Parameters/result and acceptance |
|---|---|---|
| `dataset.open` — Dataset: Open | Existing / Read; keywords `file`, `database`, `pickle` | Native file choice then analysis; cancel/error preserves active data |
| `dataset.open-recent` — Dataset: Open Recent | Existing / Read; keywords `recent`, `database` | Trusted-path quick-pick; missing entries explain removal |
| `dataset.information` — Dataset: Information | New / Read; keywords `name`, `draw count`, `date range` | Source/schema/date table; values match source without prediction preparation |
| `dataset.validate` — Dataset: Validate | New / Read; keywords `integrity`, `duplicates`, `range`, `schema` | Source validation report; never modifies the file |
| `dataset.import` — Dataset: Import Draws | New / Write; keywords `import`, `csv`, `draws` | Mapping/conflict preview and confirmation; commit is validated and atomic |
| `dataset.analyze` — Dataset: Analyze | Existing / Read; keywords `analyze`, `reports`, `statistics` | Dataset/settings and progress; failure retains prior usable analysis |
| `dataset.reanalyze` — Dataset: Reanalyze | Existing / Read; keywords `refresh`, `recalculate` | Active dataset/current settings; replace only after success |
| `dataset.export-analysis` — Dataset: Export Analysis | Existing / Write; keywords `export`, `archive`, `tables` | Destination/options; confirm overwrite and include manifest |
| `report.open` — Reports: Open | Existing / Read; keywords `workspace`, `dashboard`, `report` | Enabled-report quick-pick; navigate without recalculation |
| `report.enable` / `report.disable` — Reports: Enable/Disable | Existing / Write; keywords `plugin`, `report`, `toggle` | Multi-select/effect preview; cancel preserves settings |
| `report.export` — Reports: Export Active | New / Write; keywords `export`, `chart`, `table` | Format/destination; include filters/provenance and confirm overwrite |

### Strategies and predictions

| Command | Status/effect | Parameters/result and acceptance |
|---|---|---|
| `strategy.enable` / `strategy.disable` — Strategies: Enable/Disable | Existing / Write; keywords `strategy`, `plugin`, `toggle` | Multi-select/dependency preview; explain Hybrid internals |
| `strategy.compare` — Strategies: Compare | New / Read; keywords `compare`, `rank`, `score`, `metrics` | Strategies/range dashboard; align causal forecasts and label score scales |
| `strategy.inspect` — Strategies: Inspect | New / Read; keywords `explain`, `model`, `details` | Strategy/reference routes to commands 51--73; unsupported items explain why |
| `strategy.open-documentation` — Strategies: Open Documentation | Existing / Read; keywords `help`, `guide`, `formula` | Strategy quick-pick; use registered guide mapping |
| `prediction.generate` — Predictions: Generate | Existing / Read; keywords `predict`, `forecast`, `top 6` | Strategy/settings and progress; prefix-only forecast records provenance |
| `prediction.open` — Predictions: Open | Existing / Read; keywords `prediction`, `grid`, `forecast` | Available-suite navigation; never silently generates a suite |
| `prediction.compare` — Predictions: Compare | New / Read; keywords `comparison`, `agreement`, `strategy` | Strategies/forecast dashboard; reuse 52/68 and preserve native scales |
| `prediction.inspect` — Predictions: Inspect | Existing / Read; keywords `candidate`, `score`, `rank`, `audit` | Strategy/forecast/candidate detail; reconcile saved suite |
| `prediction.export` — Predictions: Export | New / Write; keywords `export`, `csv`, `json`, `forecast` | Suite/detail/format/destination; include provenance and confirm overwrite |

### Possible Draw and Draw Portfolio

| Command | Status/effect | Parameters/result and acceptance |
|---|---|---|
| `possible-draw.open` — Possible Draw: Open | Existing / Read; keywords `possible`, `ticket`, `draw` | Navigate and preserve current draw |
| `possible-draw.generate` — Possible Draw: Generate | Existing / Write; keywords `generate`, `ticket`, `strategies` | Strategy/settings preview; validate six distinct values and record sources |
| `possible-draw.regenerate` — Possible Draw: Regenerate | Existing / Write; keywords `again`, `replace`, `ticket` | Preselect current parameters; cancel preserves prior draw |
| `possible-draw.validate` — Possible Draw: Validate | New / Read; keywords `validate`, `constraints`, `duplicate` | Check size, uniqueness, range and constraints without mutation |
| `possible-draw.add-to-portfolio` — Possible Draw: Add to Portfolio | Existing / Write; keywords `portfolio`, `save`, `ticket` | Draw/label confirmation with explicit duplicate policy |
| `portfolio.open` — Draw Portfolio: Open | Existing / Read; keywords `portfolio`, `tickets` | Navigate and preserve selection/filters |
| `portfolio.generate` — Draw Portfolio: Generate | Existing / Write; keywords `generate`, `coverage`, `tickets` | Count/strategies/constraints preview; validate all and retain provenance |
| `portfolio.add` / `portfolio.edit` — Draw Portfolio: Add/Edit | Existing / Write; keywords `ticket`, `numbers`, `modify` | Six values/label preview; invalid input cannot replace valid state |
| `portfolio.remove` / `portfolio.clear` — Draw Portfolio: Remove/Clear | Existing / Destructive; keywords `delete`, `remove`, `clear` | Exact target preview/confirmation; state recoverability |
| `portfolio.validate` — Draw Portfolio: Validate | New / Read; keywords `validate`, `duplicate`, `coverage` | Report every invalid/duplicate ticket without mutation |
| `portfolio.export` — Draw Portfolio: Export | Existing / Write; keywords `export`, `csv`, `tickets` | Format/fields/destination; count matches and overwrite is confirmed |

### Managed draw history

| Command | Status/effect | Parameters/result and acceptance |
|---|---|---|
| `draw-history.open` — Draw History: Open | Existing / Read; keywords `history`, `managed draws` | Navigate without unrelated analysis |
| `draw-history.search` — Draw History: Search | New / Read; keywords `find`, `date`, `number` | Date/ID/number query; empty result explicit and data unchanged |
| `draw-history.add` / `draw-history.edit` — Draw History: Add/Edit | Existing / Write; keywords `draw`, `date`, `numbers`, `modify` | Date/six-value preview; enforce schema, range, uniqueness and date policy atomically |
| `draw-history.delete` — Draw History: Delete | Existing / Destructive; keywords `delete`, `remove`, `draw` | Exact row preview/confirmation; explain recovery |
| `draw-history.import` — Draw History: Import | Existing / Write; keywords `import`, `csv`, `merge` | Source/mapping/conflict preview; validate before commit |
| `draw-history.export` — Draw History: Export | New / Write; keywords `export`, `backup`, `csv` | Range/format/destination; report row count and confirm overwrite |

### Settings, navigation, and help

| Command | Status/effect | Parameters/result and acceptance |
|---|---|---|
| `settings.open` — Preferences: Open Settings | Existing / Read; keywords `settings`, `preferences` | Optional section; focus without changing values |
| `settings.search` — Preferences: Search Settings | New / Read; keywords `find`, `option`, `preference` | Fuzzy query/navigation using registered names; explicit no-match |
| `settings.reset-section` — Preferences: Reset Section | New / Destructive; keywords `reset`, `defaults`, `section` | Before/after preview; reset only named section and report invalidation |
| `settings.select-theme` / `settings.select-chart-theme` — Preferences: Select Theme | Existing / Write; keywords `appearance`, `color`, `chart`, `theme` | Preview quick-pick; cancel restores and commit persists normally |
| `view.open-workspace` — View: Open Workspace | Existing / Read; keywords `view`, `workspace`, `dashboard`, `tab` | Pick every report/tab, including Last Seen variants, Predictions, Portfolio, Possible Draw and History |
| `view.toggle-fullscreen` — View: Toggle Full Screen | Existing / Write; keywords `fullscreen`, `window`, `view` | Use existing Electron toggle behavior |
| `help.open-command-documentation` — Help: Command Documentation | New / Read; keywords `help`, `commands`, `palette`, `manual` | Open current guide or catalog; packaged target must resolve |
| `help.open-strategy-documentation` — Help: Strategy Documentation | Existing / Read; keywords `help`, `strategy`, `manual` | Pick strategy guide; missing guide disabled with reason |
| `help.about` — Help: About Rand AI | Existing / Read; keywords `about`, `version`, `license` | Show packaged application metadata |

## Proposed command interfaces

The eventual registry should extend current metadata with these documented
concepts. Names remain proposals until the first dependent command is built:

```typescript
type CommandKind = "statistics" | "strategy-diagnostic" | "navigation" | "action";
type CommandEffect = "read" | "write" | "destructive";

type CommandArgument =
  | { kind: "quick-pick"; id: string; options: readonly CommandOption[] }
  | { kind: "multi-pick"; id: string; options: readonly CommandOption[] }
  | { kind: "number"; id: string; minimum?: number; maximum?: number }
  | { kind: "number-set"; id: string; minimum: number; maximum: number }
  | { kind: "date-or-draw"; id: string };

type CommandResult =
  | EducationalStatisticsResult
  | FigureResult
  | TableResult
  | CompositeResult
  | NavigationResult
  | ActionPreviewResult;
```

Generic metadata adds `kind`, `effect`, arguments, requirements, disabled
reason and confirmation policy to ID, title, category, keywords, availability
and executor. Statistical provenance contains command ID, dataset, complete
draw count, parameters, calculation time, causal cutoff and strategy links.
Composite results own explanation, assumptions, limitations and typed sections.

Only validated IDs and typed arguments cross Electron/Python boundaries. An ID
cannot become a module, expression, file path or arbitrary bridge operation.
Navigation does not open another `BrowserWindow`; statistics retain the
renderer Esc-only overlay.

## Strategy-to-command coverage

The matrix is an implementation audit, not a claim that every historical
feature is predictive. Commands 51, 52, 72 and 73 apply to every deterministic
ranking strategy. Command 57 applies wherever production exposes a normalized
probability distribution; 61--65 apply when the target/forecast semantics make
the metric meaningful; 66--68 apply to ensembles. The table lists each
strategy's additional core statistical dependencies.

| Strategy ID | Family | Core catalog commands |
|---|---|---|
| `proximity` | Shape & Similarity | 3, 7, 13--16, 19 |
| `freshness` | Frequency & Recency | 3--6 |
| `emd` | Shape & Similarity | 19, 20 |
| `recurrence_dynamics` | Shape & Similarity | 8, 11, 12, 18, 19, 39--41 |
| `randomness` | Random Baselines | 57, 72 |
| `fresh_random` | Random Baselines | 3, 5, 57, 72 |
| `chi_square` | Frequency & Recency | 1, 28 |
| `categorical_chi_square` | Frequency & Recency | 3, 13, 29 |
| `entropy` | Frequency & Recency | 1, 5, 30 |
| `markov100` | Markov & Sequence | 3, 4, 33, 55 |
| `mkgsv` | Markov & Sequence | 3, 4, 13, 33, 55 |
| `mkfr` | Markov & Sequence | 3, 4, 34, 55 |
| `mksp` | Markov & Sequence | 13--15, 19, 35, 55 |
| `mknp` | Markov & Sequence | 7, 17, 36, 55 |
| `mkrd` | Markov & Sequence | 18, 37, 55 |
| `bayesian` | Frequency & Recency | 1, 3--5, 33, 56, 57 |
| `predictive_grid` | Shape & Similarity | 1, 3, 5, 8--10, 19--22, 32, 33, 67 |
| `co_occurrence` | Relationships & ML | 1, 21, 22, 32 |
| `doublet_triplet_markov` | Markov & Sequence | 11, 23, 24, 38, 55 |
| `mixed` | Ensembles & Coverage | 66--68 |
| `svc` | Relationships & ML | 1, 3, 5, 53, 54, 57, 58, 61--65 |
| `svc_recurrence_hybrid` | Ensembles & Coverage | 39--41, 53, 57, 61--68 |
| `svc_recurrence_proximity_hybrid` | Ensembles & Coverage | 3, 19, 39--41, 53, 57, 61--68 |
| `srph_residual_diversity_hybrid` | Ensembles & Coverage | 61--69 |
| `srph_minimax_regret_hybrid` | Ensembles & Coverage | 61--68, 70 |
| `tbl` | Relationships & ML | 1, 3, 5, 21, 53, 54, 59 |
| `sklearn_svm` | Relationships & ML | 1, 3, 5, 53, 54, 57, 58, 61--65 |
| `lag_logistic` | Relationships & ML | 1, 3--5, 53, 54, 57, 59, 61--65 |
| `sparse_neural_ticket` | Relationships & ML | 1, 3, 5, 21, 53, 54, 57, 61--65 |
| `cis` | Ensembles & Coverage | 66--68 |
| `decision_tree_selector` | Ensembles & Coverage | 53, 54, 60, 64, 65 |
| `border_group_statistical` | Border Space Groups | 2, 42, 43, 57 |
| `border_group_markov` | Border Space Groups | 2, 42, 44--46, 55, 57 |
| `border_group_bayesian` | Border Space Groups | 2, 42, 44, 56, 57 |
| `border_group_ml` | Border Space Groups | 2, 13, 16, 47--50, 53, 54, 57, 59, 61--65 |
| `border_group_svc` | Border Space Groups | 2, 13, 16, 47--50, 53, 54, 57, 58, 61--65 |
| `border_group_hybrid` | Border Space Groups | 2, 42--50, 55--67 |
| `residual_coverage` | Ensembles & Coverage | 68, 69, 71 |
| `chained` | Ensembles & Coverage | 66, 67, 71 |

Border Group SVC is intentionally present even while its catalog/registration
changes are being integrated, so implementing its diagnostic commands cannot
silently omit it. Its row covers lagged signatures, group counts, maximum and
current spaces, 10/25/100-draw summaries, class balance, fitted probabilities,
support vectors and walk-forward metrics.

## Documentation validation

A documentation change implementing or reordering a proposal must:

1. keep numbered statistics/diagnostic headings contiguous from 1 through 73;
2. keep all command identifiers unique, including IDs combined in operational
   rows;
3. audit the strategy matrix against the registered `StrategyId` catalog and
   include Border Group SVC;
4. audit report/workspace actions against registered report and workspace IDs;
5. preserve the distinction between implemented and proposed commands;
6. keep every entry's inputs, output, consumers, failure behavior and acceptance
   statement complete; and
7. build Sphinx with warnings treated as errors.

## Assumptions and implementation order

This page changes documentation only. Proposed identifiers and interfaces are
stable backlog targets but do not become public runtime API until implemented.
Commands should normally be implemented in numeric order; an entry marked
**Dependent** waits for all named prerequisites. Shared concepts receive one
reusable command rather than strategy-specific duplicates.

Statistics use the complete active database by default. Strategy diagnostics
use a causal replay whenever training or forecast state is involved. Quick-picks
collect parameters, educational results use the full-renderer Esc-only overlay,
and write/destructive actions require confirmation. Existing commands 1 and 2
retain their present identifiers and behavior.
