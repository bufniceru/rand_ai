# Statistics and prediction simplification study

Exploratory historical evidence only: this dataset previously influenced strategy selection. The last two blocks are a frozen within-study comparison, not independent confirmation.

Reproduce: `uv run python scripts/study_statistics.py`

Dataset SHA-256: `e46d899b32e8056580d68c86d7c7a623d96e25317e5469e1613600f531ca7a2b`. Draws: 771.
Targets by block: [[121, 251], [252, 381], [382, 511], [512, 641], [642, 771]]. Random expectation: 0.734694 hits/draw.
Protocol and raw rankings, selections, hits, dependencies, export fields, and uncertainty are in the adjacent JSON/JSONL/CSV files.
Moving-block bootstrap: length 20, 10,000 replicates, seed 20261006. One-sided centered bootstrap tests use Holm correction across all baseline, ablation, and merge improvement claims. Equivalence uses paired 95% intervals; it is a simplification screen, not proof of future equivalence.

## Strategies

| Strategy | Draws | Hits | Mean | 3+ rate | Five block means | Decision |
|---|---:|---:|---:|---:|---|---|
| recurrence_dynamics | 651 | 541 | 0.8310 | 3.072% | 0.878, 0.915, 0.769, 0.769, 0.823 | retain experimental; no corrected improvement over both random baselines |
| srph_residual_diversity_hybrid | 651 | 534 | 0.8203 | 1.690% | 0.817, 0.931, 0.738, 0.777, 0.838 | consolidate default presentation into svc_recurrence_proximity_hybrid; retain backend |
| svc_recurrence_proximity_hybrid | 651 | 534 | 0.8203 | 1.843% | 0.817, 0.900, 0.731, 0.746, 0.908 | retain experimental; no corrected improvement over both random baselines |
| svc_recurrence_hybrid | 651 | 530 | 0.8141 | 2.151% | 0.786, 0.885, 0.754, 0.777, 0.869 | retain experimental; no corrected improvement over both random baselines |
| srph_minimax_regret_hybrid | 651 | 526 | 0.8080 | 1.997% | 0.802, 0.915, 0.731, 0.777, 0.815 | consolidate default presentation into svc_recurrence_proximity_hybrid; retain backend |
| svc | 651 | 524 | 0.8049 | 1.843% | 0.740, 0.823, 0.923, 0.800, 0.738 | retain experimental; no corrected improvement over both random baselines |
| decision_tree_selector | 651 | 513 | 0.7880 | 2.304% | 0.786, 0.854, 0.715, 0.838, 0.746 | retain experimental; no corrected improvement over both random baselines |
| sparse_neural_ticket | 250 | 197 | 0.7880 | 2.400% | unavailable, unavailable, unavailable, 0.758, 0.815 | retain experimental; no corrected improvement over both random baselines |
| categorical_chi_square | 651 | 509 | 0.7819 | 2.611% | 0.809, 0.862, 0.715, 0.785, 0.738 | retain experimental; no corrected improvement over both random baselines |
| doublet_triplet_markov | 651 | 509 | 0.7819 | 2.611% | 0.702, 0.754, 0.808, 0.823, 0.823 | retain experimental; no corrected improvement over both random baselines |
| chi_square | 651 | 507 | 0.7788 | 2.151% | 0.817, 0.915, 0.731, 0.754, 0.677 | consolidate default presentation into categorical_chi_square; retain backend |
| bayesian | 651 | 503 | 0.7727 | 1.843% | 0.802, 0.892, 0.677, 0.731, 0.762 | retain experimental; no corrected improvement over both random baselines |
| emd | 651 | 503 | 0.7727 | 1.843% | 0.824, 0.838, 0.685, 0.846, 0.669 | retain experimental; no corrected improvement over both random baselines |
| mksp | 651 | 502 | 0.7711 | 1.382% | 0.809, 0.792, 0.708, 0.792, 0.754 | retain experimental; no corrected improvement over both random baselines |
| sklearn_svm | 651 | 500 | 0.7680 | 2.304% | 0.672, 0.762, 0.854, 0.823, 0.731 | retain experimental; no corrected improvement over both random baselines |
| tbl | 651 | 500 | 0.7680 | 1.229% | 0.725, 0.746, 0.723, 0.754, 0.892 | retain experimental; no corrected improvement over both random baselines |
| proximity | 651 | 499 | 0.7665 | 2.151% | 0.817, 0.900, 0.715, 0.731, 0.669 | retain experimental; no corrected improvement over both random baselines |
| fresh_random | 651 | 498 | 0.7650 | 2.765% | 0.718, 0.785, 0.654, 0.846, 0.823 | retain experimental; no corrected improvement over both random baselines |
| chained | 651 | 496 | 0.7619 | 2.151% | 0.702, 0.877, 0.792, 0.700, 0.738 | retain experimental; no corrected improvement over both random baselines |
| lag_logistic | 651 | 496 | 0.7619 | 1.997% | 0.748, 0.815, 0.754, 0.785, 0.708 | retain experimental; no corrected improvement over both random baselines |
| mixed | 651 | 491 | 0.7542 | 2.304% | 0.748, 0.792, 0.777, 0.723, 0.731 | retain experimental; no corrected improvement over both random baselines |
| entropy | 651 | 490 | 0.7527 | 1.536% | 0.817, 0.731, 0.800, 0.708, 0.708 | retain experimental; no corrected improvement over both random baselines |
| border_group_bayesian | 651 | 489 | 0.7512 | 1.536% | 0.802, 0.808, 0.762, 0.685, 0.700 | retain experimental; no corrected improvement over both random baselines |
| mkfr | 651 | 488 | 0.7496 | 2.151% | 0.809, 0.738, 0.692, 0.800, 0.708 | retain experimental; no corrected improvement over both random baselines |
| predictive_grid | 651 | 488 | 0.7496 | 1.997% | 0.779, 0.846, 0.677, 0.785, 0.662 | retain experimental; no corrected improvement over both random baselines |
| randomness | 651 | 488 | 0.7496 | 1.997% | 0.756, 0.885, 0.654, 0.677, 0.777 | retain experimental; no corrected improvement over both random baselines |
| mknp | 651 | 487 | 0.7481 | 1.843% | 0.802, 0.715, 0.762, 0.685, 0.777 | retain experimental; no corrected improvement over both random baselines |
| border_group_hybrid | 651 | 486 | 0.7465 | 1.229% | 0.840, 0.762, 0.731, 0.692, 0.708 | consolidate default presentation into border_group_bayesian; retain backend |
| mkrd | 651 | 483 | 0.7419 | 1.997% | 0.786, 0.746, 0.777, 0.638, 0.762 | consolidate default presentation into mknp; retain backend |
| markov100 | 651 | 481 | 0.7389 | 2.919% | 0.664, 0.608, 0.785, 0.831, 0.808 | retain experimental; no corrected improvement over both random baselines |
| mkgsv | 651 | 481 | 0.7389 | 2.919% | 0.664, 0.608, 0.785, 0.831, 0.808 | consolidate default presentation into markov100; retain backend |
| border_group_markov | 651 | 480 | 0.7373 | 1.229% | 0.802, 0.738, 0.754, 0.692, 0.700 | consolidate default presentation into border_group_bayesian; retain backend |
| border_group_statistical | 651 | 475 | 0.7296 | 1.075% | 0.817, 0.731, 0.723, 0.677, 0.700 | retain experimental; no corrected improvement over both random baselines |
| border_group_svc | 651 | 475 | 0.7296 | 1.690% | 0.824, 0.731, 0.715, 0.677, 0.700 | consolidate default presentation into border_group_statistical; retain backend |
| cis | 651 | 473 | 0.7266 | 1.690% | 0.740, 0.769, 0.731, 0.631, 0.762 | retain experimental; no corrected improvement over both random baselines |
| co_occurrence | 651 | 473 | 0.7266 | 1.536% | 0.832, 0.746, 0.646, 0.785, 0.623 | retain experimental; no corrected improvement over both random baselines |
| border_group_ml | 651 | 467 | 0.7174 | 1.536% | 0.702, 0.762, 0.662, 0.769, 0.692 | retain experimental; no corrected improvement over both random baselines |
| freshness | 651 | 466 | 0.7158 | 1.536% | 0.672, 0.592, 0.785, 0.785, 0.746 | retain experimental; no corrected improvement over both random baselines |
| residual_coverage | 651 | 415 | 0.6375 | 1.229% | 0.603, 0.623, 0.646, 0.700, 0.615 | retain experimental; no corrected improvement over both random baselines |
| seeded_random | 651 | 469 | 0.7204 | 2.304% | 0.702, 0.746, 0.754, 0.731, 0.669 | fixed independent random comparison |

sparse_neural_ticket has a shorter evaluation period: targets [522, 771], 401 unavailable forecasts. Its mean is not comparable to full-horizon means.


## Redundancy

- border_group_bayesian / border_group_hybrid: overlap 5.284; paired interval [-0.027649769585253458, 0.03533026113671275]; consolidate: True.
- border_group_bayesian / border_group_markov: overlap 5.247; paired interval [-0.013824884792626729, 0.041474654377880185]; consolidate: True.
- border_group_bayesian / border_group_statistical: overlap 5.117; paired interval [-0.009216589861751152, 0.05222734254992319]; consolidate: False.
- border_group_bayesian / border_group_svc: overlap 5.088; paired interval [-0.009216589861751152, 0.05222734254992319]; consolidate: False.
- border_group_hybrid / border_group_markov: overlap 5.799; paired interval [-0.004608294930875576, 0.02457757296466974]; consolidate: True.
- border_group_hybrid / border_group_statistical: overlap 5.739; paired interval [0.004608294930875576, 0.03225806451612903]; consolidate: True.
- border_group_hybrid / border_group_svc: overlap 5.662; paired interval [0.0030721966205837174, 0.03379416282642089]; consolidate: True.
- border_group_markov / border_group_statistical: overlap 5.685; paired interval [-0.009216589861751152, 0.026113671274961597]; consolidate: True.
- border_group_markov / border_group_svc: overlap 5.601; paired interval [-0.010752688172043012, 0.029185867895545316]; consolidate: True.
- border_group_statistical / border_group_svc: overlap 5.820; paired interval [-0.01228878648233487, 0.013824884792626729]; consolidate: True.
- categorical_chi_square / chi_square: overlap 5.241; paired interval [-0.027649769585253458, 0.03533026113671275]; consolidate: True.
- freshness / markov100: overlap 5.183; paired interval [-0.053763440860215055, 0.007680491551459293]; consolidate: False.
- freshness / mkgsv: overlap 5.183; paired interval [-0.053763440860215055, 0.007680491551459293]; consolidate: False.
- markov100 / mkgsv: overlap 6.000; paired interval [0.0, 0.0]; consolidate: True.
- mknp / mkrd: overlap 5.621; paired interval [-0.016897081413210446, 0.029185867895545316]; consolidate: True.
- srph_minimax_regret_hybrid / srph_residual_diversity_hybrid: overlap 5.252; paired interval [-0.043010752688172046, 0.018433179723502304]; consolidate: True.
- srph_minimax_regret_hybrid / svc_recurrence_proximity_hybrid: overlap 5.015; paired interval [-0.043010752688172046, 0.02304147465437788]; consolidate: True.
- srph_residual_diversity_hybrid / svc_recurrence_proximity_hybrid: overlap 5.479; paired interval [-0.021505376344086023, 0.027649769585253458]; consolidate: True.

Default comparison charts use the following representatives. Original methods remain in the full ranking and in expandable chart controls; enablement and saved settings do not change. Every pair within a group must qualify: equivalence is not assumed to be transitive. Representatives use highest observed mean, then fewer declared dependencies, then identifier.

| Representative | Expandable alternatives |
|---|---|
| border_group_bayesian | border_group_hybrid, border_group_markov |
| border_group_statistical | border_group_svc |
| categorical_chi_square | chi_square |
| markov100 | mkgsv |
| mknp | mkrd |
| svc_recurrence_proximity_hybrid | srph_minimax_regret_hybrid, srph_residual_diversity_hybrid |

## Input removal experiments

Families are removed at both training and prediction time; each model is retrained from scratch. SVC, TBL and online SVM are covered. Other models' inputs are untested and retained.

| Consumer / removed family | Mean | Difference | Paired 95% interval | Holm p | Decision |
|---|---:|---:|---|---:|---|
| svc / composition | 0.7435 | -0.0614 | [-0.11981566820276497, 0.0030721966205837174] | 1.0000 | retain input; removal has no corrected practical improvement |
| svc / freshness | 0.7957 | -0.0092 | [-0.053763440860215055, 0.02304147465437788] | 1.0000 | retain input; removal has no corrected practical improvement |
| svc / frequency | 0.7788 | -0.0261 | [-0.07987711213517665, 0.021505376344086023] | 1.0000 | retain input; removal has no corrected practical improvement |
| tbl / composition | 0.7742 | +0.0061 | [-0.03379416282642089, 0.055299539170506916] | 1.0000 | retain input; removal has no corrected practical improvement |
| tbl / freshness | 0.7742 | +0.0061 | [-0.0629800307219662, 0.07834101382488479] | 1.0000 | retain input; removal has no corrected practical improvement |
| tbl / frequency | 0.7742 | +0.0061 | [-0.02304147465437788, 0.03840245775729647] | 1.0000 | retain input; removal has no corrected practical improvement |
| tbl / relationships | 0.7634 | -0.0046 | [-0.021505376344086023, 0.016897081413210446] | 1.0000 | retain input; removal has no corrected practical improvement |
| tbl / expert_ranks | 0.7465 | -0.0215 | [-0.059907834101382486, 0.030721966205837174] | 1.0000 | retain input; removal has no corrected practical improvement |
| sklearn_svm / composition | 0.7650 | -0.0031 | [-0.059907834101382486, 0.055299539170506916] | 1.0000 | retain input; removal has no corrected practical improvement |
| sklearn_svm / freshness | 0.7696 | +0.0015 | [-0.059907834101382486, 0.06144393241167435] | 1.0000 | retain input; removal has no corrected practical improvement |
| sklearn_svm / frequency | 0.7650 | -0.0031 | [-0.04608294930875576, 0.043010752688172046] | 1.0000 | retain input; removal has no corrected practical improvement |
| sklearn_svm / relationships | 0.7465 | -0.0215 | [-0.055299539170506916, 0.007680491551459293] | 1.0000 | retain input; removal has no corrected practical improvement |
| sklearn_svm / expert_ranks | 0.7343 | -0.0338 | [-0.12135176651305683, 0.030721966205837174] | 1.0000 | retain input; removal has no corrected practical improvement |

## Merge experiments

- co_occurrence + sklearn_svm: confirmation mean 0.8000; parent comparisons [{'difference': 0.09615384615384616, 'ci95': [-0.0038461538461538464, 0.22692307692307692], 'p': 0.0887911208879112, 'holmP': 1.0}, {'difference': 0.023076923076923078, 'ci95': [-0.1, 0.1], 'p': 0.19408059194080593, 'holmP': 1.0}]; do not promote; retain parents.
- emd + sklearn_svm: confirmation mean 0.7769; parent comparisons [{'difference': 0.019230769230769232, 'ci95': [-0.057692307692307696, 0.11538461538461539], 'p': 0.3678632136786321, 'holmP': 1.0}, {'difference': 0.0, 'ci95': [-0.1, 0.08846153846153847], 'p': 0.4774522547745225, 'holmP': 1.0}]; do not promote; retain parents.
- entropy + randomness: confirmation mean 0.7577; parent comparisons [{'difference': 0.05, 'ci95': [-0.046153846153846156, 0.1576923076923077], 'p': 0.22357764223577642, 'holmP': 1.0}, {'difference': 0.03076923076923077, 'ci95': [-0.05384615384615385, 0.13076923076923078], 'p': 0.3454654534546545, 'holmP': 1.0}]; do not promote; retain parents.

## Statistics and presentation

Circular-space position charts and their combined heatmap share one expandable detail panel under the aggregate distribution. These are views of the same counts, rather than independent predictive signals. Summary and number/space descriptive tables become expandable details. Charts, exports, settings and diagnostic access remain available.
Drop the summed hit total and summed hit-distribution row across strategies: they count the same draw multiple times and are not the performance of a six-number prediction. Replace the aggregate card with theoretical random expectation. Keep all seven per-strategy match counts (including zero), display each strategy's actual evaluated-draw count, and exclude unavailable forecasts from averages and timeline curves.

| Statistic | Role | Fields | Decision | UI |
|---|---|---|---|---|
| summary | descriptive | metric, value | retain accessible; no standalone predictive claim | Expandable descriptive table |
| number_frequencies | descriptive | number, count, appearance_rate, observation_percentage, expected_count, deviation, standardized_residual | retain accessible; no standalone predictive claim | keep |
| position_frequencies | descriptive | position, number, count, appearance_rate | retain accessible; no standalone predictive claim | keep |
| number_descriptive | descriptive | variable, count, mean, std, min, q25, median, q75, max | retain accessible; no standalone predictive claim | Expandable descriptive table |
| freshness_gap_distribution | descriptive | gap, hits, opportunities, hit_rate, hit_percentage, expected_hits, hit_difference, expected_hit_rate, hit_rate_difference_pp | retain accessible; no standalone predictive claim | keep |
| draw_structure_distributions | descriptive | measure, value, count, percentage | retain accessible; no standalone predictive claim | keep |
| pair_cooccurrence | descriptive | number_a, number_b, count, expected_count, lift | retain accessible; no standalone predictive claim | keep |
| space_frequencies | descriptive | position, space, count, percentage | retain accessible; no standalone predictive claim | Shared expandable circular-space panel |
| distance_frequencies | descriptive | distance, occurrences, occurrence_percentage | retain accessible; no standalone predictive claim | keep |
| space_descriptive | descriptive | variable, count, mean, std, min, q25, median, q75, max | retain accessible; no standalone predictive claim | Expandable descriptive table |
| space_extreme_distributions | descriptive | measure, value, count, percentage | retain accessible; no standalone predictive claim | keep |
| number_correlations_pearson | diagnostic | num1, num2, num3, num4, num5, num6 | retain accessible; no standalone predictive claim | keep |
| space_correlations_pearson | diagnostic | dist1, dist2, dist3, dist4, dist5, dist6 | retain accessible; no standalone predictive claim | keep |
| number_space_correlations_pearson | diagnostic | dist1, dist2, dist3, dist4, dist5, dist6 | retain accessible; no standalone predictive claim | keep |
| number_correlations_spearman | diagnostic | num1, num2, num3, num4, num5, num6 | retain accessible; no standalone predictive claim | keep |
| space_correlations_spearman | diagnostic | dist1, dist2, dist3, dist4, dist5, dist6 | retain accessible; no standalone predictive claim | keep |
| number_space_correlations_spearman | diagnostic | dist1, dist2, dist3, dist4, dist5, dist6 | retain accessible; no standalone predictive claim | keep |
| number_trends | descriptive | bin, start_draw, end_draw, number, count, appearance_rate | retain accessible; no standalone predictive claim | keep |
| randomness_diagnostics | diagnostic | diagnostic, value, reference, reliable | retain accessible; no standalone predictive claim | keep |
| sampled_spaces | descriptive | position, space | retain accessible; no standalone predictive claim | keep |
| group_count_frequencies | descriptive | group_count, count | retain accessible; no standalone predictive claim | keep |
| group_signature_frequencies | descriptive | group_count, signature, count | retain accessible; no standalone predictive claim | keep |

## Auxiliary panels and on-demand commands

- overview: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- numbers: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- spaces: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- space-groups: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- relationships: diagnostic; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- randomness: diagnostic; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- nonlinear-dynamics: diagnostic; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- autocorrelation: diagnostic; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- co-occurrence: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- prediction-audit: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- draw-comparison: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- strategy-effectiveness: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- strategy-hit-statistics: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- gaps: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- last-seen: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- last-seen-gap: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- last-seen-space: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- predictions: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- draw-portfolio: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- possible-draw: prediction/evaluation; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- statistics.number-frequency: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- statistics.group-frequency: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift
- statistics.gap-statistics: descriptive; retain access and settings. Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift

Frequency count, appearance rate, observation share, deviation and standardized residual are transformations of the same observed count given the dataset size. Keep the existing single on-demand frequency chart and hover information; preserve exported fields. Space-position bars and the heatmap encode identical counts; the aggregate sums those position counts. Same-date draws retain their YAML source order.


## Cost and limitations

Shared strategy evaluation took 395.07 seconds on this machine. Ablation runtimes are recorded individually; shared computation prevents attributing the full-run time to individual strategies. Feature masks test specific consumers, not every use of a statistic. No uncertain statistical input is deleted, and no historical win is presented as proven future prediction.
