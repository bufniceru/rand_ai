# Portfolio coverage study

Exploratory historical evidence; dataset previously used for strategy selection. Synthetic runs diagnose leakage and false signals, not predictive improvement.

Stage: smoke. Seed: 20261017. No UI defaults changed.

Coverage-first is a greedy heuristic, not a minimum-wheel solver. Guarantees apply only to complete verified pool-triple coverage.

Random global coverage uses the 49-number universe; random-pool and model coverage use the selected pool. These denominators must not be conflated.

| Job | Budget | Objective | Method | Targets | Mean best hits | At least one 3+ | Exact triple coverage |
|---|---:|---|---|---:|---:|---:|---:|
| historical | 10 | blended | blended | 30 | 1.7667 | 6.67% | 15.81% |
| historical | 10 | blended | random_global | 30 | 1.9877 | 17.57% | 1.08% |
| historical | 10 | blended | random_pool | 30 | 1.7017 | 9.20% | 16.23% |
| historical | 10 | coverage_first | coverage_first | 30 | 1.7000 | 3.33% | 17.54% |
| historical | 10 | coverage_first | random_global | 30 | 1.9877 | 17.57% | 1.08% |
| historical | 10 | coverage_first | random_pool | 30 | 1.7017 | 9.20% | 16.23% |
| synthetic-00 | 10 | blended | blended | 30 | 1.9000 | 30.00% | 15.85% |
| synthetic-00 | 10 | blended | random_global | 30 | 1.9950 | 16.93% | 1.08% |
| synthetic-00 | 10 | blended | random_pool | 30 | 1.7980 | 15.33% | 16.23% |
| synthetic-00 | 10 | coverage_first | coverage_first | 30 | 1.9333 | 30.00% | 17.54% |
| synthetic-00 | 10 | coverage_first | random_global | 30 | 1.9950 | 16.93% | 1.08% |
| synthetic-00 | 10 | coverage_first | random_pool | 30 | 1.7980 | 15.33% | 16.23% |
| synthetic-01 | 10 | blended | blended | 30 | 1.5000 | 0.00% | 16.26% |
| synthetic-01 | 10 | blended | random_global | 30 | 1.9933 | 17.80% | 1.08% |
| synthetic-01 | 10 | blended | random_pool | 30 | 1.5980 | 8.20% | 16.22% |
| synthetic-01 | 10 | coverage_first | coverage_first | 30 | 1.5333 | 0.00% | 17.54% |
| synthetic-01 | 10 | coverage_first | random_global | 30 | 1.9933 | 17.80% | 1.08% |
| synthetic-01 | 10 | coverage_first | random_pool | 30 | 1.5980 | 8.20% | 16.22% |
| synthetic-02 | 10 | blended | blended | 30 | 2.0000 | 30.00% | 16.04% |
| synthetic-02 | 10 | blended | random_global | 30 | 1.9633 | 16.63% | 1.08% |
| synthetic-02 | 10 | blended | random_pool | 30 | 1.8663 | 16.27% | 16.22% |
| synthetic-02 | 10 | coverage_first | coverage_first | 30 | 2.0000 | 23.33% | 17.54% |
| synthetic-02 | 10 | coverage_first | random_global | 30 | 1.9633 | 16.63% | 1.08% |
| synthetic-02 | 10 | coverage_first | random_pool | 30 | 1.8663 | 16.27% | 16.22% |

## Paired comparisons

Positive differences favor the first method. Intervals are paired moving-block bootstrap intervals; Holm adjustment covers all primary comparisons within historical/synthetic families. Short smoke histories are preliminary.

| Job/budget | Comparison | Metric | Difference | 95% interval | Holm p |
|---|---|---|---:|---|---:|
| historical/10 | blended/blended vs blended/random_global | bestHits | -0.2210 | [-0.3163333333333333, -0.1813333333333333] | 1.0000 |
| historical/10 | blended/blended vs blended/random_pool | bestHits | 0.0650 | [-0.018666666666666682, 0.08233333333333333] | 0.0010 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2877 | [-0.3783333333333332, -0.18399999999999997] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | -0.0017 | [-0.031333333333333345, 0.04499999999999998] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs blended/blended | bestHits | -0.0667 | [-0.06666666666666667, 0.03333333333333333] | 1.0000 |
| historical/10 | blended/blended vs blended/random_global | threePlus | -0.1090 | [-0.14166666666666666, -0.07366666666666669] | 1.0000 |
| historical/10 | blended/blended vs blended/random_pool | threePlus | -0.0253 | [-0.05033333333333333, 0.003000000000000002] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.1423 | [-0.1766666666666667, -0.136] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0587 | [-0.09066666666666667, -0.03666666666666667] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0333 | [-0.06666666666666667, 0.0] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_global | bestHits | -0.0950 | [-0.3583333333333332, 0.0453333333333334] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_pool | bestHits | 0.1020 | [-0.026000000000000023, 0.2643333333333333] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.0617 | [-0.39101666666666657, 0.044000000000000074] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.1353 | [-0.07600000000000001, 0.24199999999999994] | 0.0030 |
| synthetic-00/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0333 | [-0.13333333333333333, 0.06666666666666667] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_global | threePlus | 0.1307 | [0.031666666666666676, 0.2037666666666555] | 0.0030 |
| synthetic-00/10 | blended/blended vs blended/random_pool | threePlus | 0.1467 | [0.095, 0.21766666666666667] | 0.0030 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | 0.1307 | [-0.0016666666666666692, 0.16933333333333334] | 0.0030 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.1467 | [0.05466666666666668, 0.17833333333333334] | 0.0030 |
| synthetic-00/10 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0000 | [-0.06666666666666667, 0.0] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_global | bestHits | -0.4933 | [-0.5600333333333326, -0.26133333333333325] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_pool | bestHits | -0.0980 | [-0.12799999999999997, -0.052666666666666646] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.4600 | [-0.5206666666666666, -0.3273333333333333] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | -0.0647 | [-0.18966666666666665, -0.03399999999999998] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0333 | [-0.06666666666666667, 0.06666666666666667] | 0.7739 |
| synthetic-01/10 | blended/blended vs blended/random_global | threePlus | -0.1780 | [-0.17999999999999997, -0.17200000000000004] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_pool | threePlus | -0.0820 | [-0.12166666666666667, -0.07200000000000001] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.1780 | [-0.18000000000000002, -0.17200000000000004] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0820 | [-0.12166666666666667, -0.07200000000000001] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0000 | [0.0, 0.0] | 1.0000 |
| synthetic-02/10 | blended/blended vs blended/random_global | bestHits | 0.0367 | [-0.1739999999999999, 0.09000000000000004] | 0.7739 |
| synthetic-02/10 | blended/blended vs blended/random_pool | bestHits | 0.1337 | [0.031333333333333324, 0.1716666666666667] | 0.0030 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | 0.0367 | [-0.16999999999999993, 0.05733333333333339] | 0.2900 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.1337 | [0.029000000000000005, 0.19866666666666663] | 0.0030 |
| synthetic-02/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0000 | [-0.06666666666666667, 0.1] | 1.0000 |
| synthetic-02/10 | blended/blended vs blended/random_global | threePlus | 0.1337 | [0.06832500000000002, 0.19466666666666668] | 0.0030 |
| synthetic-02/10 | blended/blended vs blended/random_pool | threePlus | 0.1373 | [0.10466666666666669, 0.21533333333333335] | 0.0030 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | 0.0670 | [-0.003000000000000002, 0.12699999999999997] | 0.0030 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0707 | [0.036666666666666674, 0.13166666666666668] | 0.4313 |
| synthetic-02/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0667 | [-0.1, -0.03333333333333333] | 1.0000 |

## Runtime and availability

| Job/budget | Objective | Generation ms/target | Model-weighted triple coverage | Mean uncovered triples | Guaranteed targets |
|---|---|---:|---:|---:|---:|
| historical/10 | blended | 90.25 | 0.20397722981234617 | 959.7666666666667 | 0 |
| historical/10 | coverage_first | 50.36 | 0.21266599248551435 | 940.0 | 0 |
| synthetic-00/10 | blended | 99.14 | 0.2038230958889972 | 959.3 | 0 |
| synthetic-00/10 | coverage_first | 57.29 | 0.21011586391588205 | 940.0 | 0 |
| synthetic-01/10 | blended | 89.98 | 0.1906290314133445 | 954.6666666666666 | 0 |
| synthetic-01/10 | coverage_first | 51.32 | 0.19835877890686732 | 940.0 | 0 |
| synthetic-02/10 | blended | 89.59 | 0.19744852561373213 | 957.2 | 0 |
| synthetic-02/10 | coverage_first | 51.76 | 0.2054600825614211 | 940.0 | 0 |
- historical/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 3044.4604999999997}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 1816.4240999999997}]; unavailable forecast counts none.
- synthetic-00/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 3320.3933}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 2044.9429}]; unavailable forecast counts none.
- synthetic-01/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 3029.9051}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 1859.3409000000001}]; unavailable forecast counts none.
- synthetic-02/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 3009.5566999999996}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 1862.2112999999995}]; unavailable forecast counts none.

## Recommendation

Keep coverage-first research-only. This smoke run validates execution and reports preliminary diagnostics; complete historical and synthetic stages before deciding on UI exposure.

Screening defaults: positive exact coverage gain at every budget; paired lower bounds at least −0.05 best hits and −0.01 three-plus rate; coverage-first generation at most 2 seconds per target. No predictive-superiority claim without independent future evidence.
