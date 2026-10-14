# XGBoost chronological study

Exploratory; history previously used for strategy development.

Random expectation: 0.734694 hits. Runtime: 16.21s.

| Strategy | Draws | Mean hits | 3+ rate |
|---|---:|---:|---:|
| xgboost_classifier | 650 | 0.716923 | 1.85% |
| lag_logistic | 650 | 0.761538 | 2.00% |
| smoothed_frequency | 650 | 0.786154 | 2.15% |
| randomness | 650 | 0.750769 | 2.00% |

## Paired differences (XGBoost minus comparator)

- lag_logistic: -0.044615; 95% moving-block bootstrap interval [-0.14, 0.033846153846153845].
- smoothed_frequency: -0.069231; 95% moving-block bootstrap interval [-0.13692307692307693, -0.0015384615384615385].
- randomness: -0.033846; 95% moving-block bootstrap interval [-0.1, 0.03538461538461538].

Full hit distributions and five chronological block means are in xgboost_study.json.
