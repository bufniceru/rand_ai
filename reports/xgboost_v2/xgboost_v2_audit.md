# XGBoost V2 audit

Exploratory historical evidence; repeated gates and inspected history do not establish future advantage.

Evaluation: retrospective. Fallback: 92.5%.

| Strategy | Draws | Mean hits | Log loss | Brier |
|---|---:|---:|---:|---:|
| xgboost_classifier | 530 | 0.726415 | 0.372078 | 0.107523 |
| raw_v2 | 530 | 0.698113 | 0.371876 | 0.107478 |
| xgboost_classifier_v2 | 530 | 0.762264 | 0.372614 | 0.107633 |
| lag_logistic | 530 | 0.758491 | 0.374445 | 0.107953 |
| smoothed_frequency | 530 | 0.773585 | 0.372769 | 0.107666 |
| uniform | 530 | 0.756604 | 0.371776 | 0.107455 |

## Learning controls

{"passed": true, "fairApparentImprovements": 0, "interpretation": "Fixed 500-transition training / 499-transition test; all eight configurations reported; no test-selected winner."}

Full paired intervals, five chronological blocks, configuration diagnostics, and control results are in xgboost_v2_audit.json.

Runtime: 138.33s. No configuration or gate thresholds were changed based on these results.
