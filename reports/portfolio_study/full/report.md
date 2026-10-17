# Portfolio coverage study

Exploratory historical evidence; dataset previously used for strategy selection. Synthetic runs diagnose leakage and false signals, not predictive improvement.

Stage: all. Seed: 20261017. No UI defaults changed.

Coverage-first is a greedy heuristic, not a minimum-wheel solver. Guarantees apply only to complete verified pool-triple coverage.

Random global coverage uses the 49-number universe; random-pool and model coverage use the selected pool. These denominators must not be conflated.

| Job | Budget | Objective | Method | Targets | Mean best hits | At least one 3+ | Exact triple coverage |
|---|---:|---|---|---:|---:|---:|---:|
| historical | 5 | blended | blended | 651 | 1.5100 | 9.22% | 12.03% |
| historical | 5 | blended | random_global | 651 | 1.6412 | 9.15% | 0.54% |
| historical | 5 | blended | random_pool | 651 | 1.4720 | 8.69% | 11.67% |
| historical | 5 | coverage_first | coverage_first | 651 | 1.4931 | 8.60% | 12.25% |
| historical | 5 | coverage_first | random_global | 651 | 1.6412 | 9.15% | 0.54% |
| historical | 5 | coverage_first | random_pool | 651 | 1.4720 | 8.69% | 11.67% |
| historical | 10 | blended | blended | 651 | 1.7112 | 13.82% | 15.63% |
| historical | 10 | blended | random_global | 651 | 1.9865 | 17.18% | 1.08% |
| historical | 10 | blended | random_pool | 651 | 1.7592 | 15.01% | 16.22% |
| historical | 10 | coverage_first | coverage_first | 651 | 1.8018 | 16.59% | 17.54% |
| historical | 10 | coverage_first | random_global | 651 | 1.9865 | 17.18% | 1.08% |
| historical | 10 | coverage_first | random_pool | 651 | 1.7592 | 15.01% | 16.22% |
| historical | 20 | blended | blended | 651 | 1.8848 | 19.20% | 20.07% |
| historical | 20 | blended | random_global | 651 | 2.2956 | 31.23% | 2.15% |
| historical | 20 | blended | random_pool | 651 | 2.0261 | 24.95% | 23.01% |
| historical | 20 | coverage_first | coverage_first | 651 | 2.0814 | 27.80% | 25.97% |
| historical | 20 | coverage_first | random_global | 651 | 2.2956 | 31.23% | 2.15% |
| historical | 20 | coverage_first | random_pool | 651 | 2.0261 | 24.95% | 23.01% |
| synthetic-00 | 5 | blended | blended | 240 | 1.4625 | 7.92% | 12.18% |
| synthetic-00 | 5 | blended | random_global | 240 | 1.6452 | 9.13% | 0.54% |
| synthetic-00 | 5 | blended | random_pool | 240 | 1.4254 | 7.82% | 11.67% |
| synthetic-00 | 5 | coverage_first | coverage_first | 240 | 1.4458 | 9.58% | 12.25% |
| synthetic-00 | 5 | coverage_first | random_global | 240 | 1.6452 | 9.13% | 0.54% |
| synthetic-00 | 5 | coverage_first | random_pool | 240 | 1.4254 | 7.82% | 11.67% |
| synthetic-00 | 10 | blended | blended | 240 | 1.6750 | 15.83% | 16.05% |
| synthetic-00 | 10 | blended | random_global | 240 | 1.9897 | 17.26% | 1.08% |
| synthetic-00 | 10 | blended | random_pool | 240 | 1.7170 | 13.32% | 16.22% |
| synthetic-00 | 10 | coverage_first | coverage_first | 240 | 1.7333 | 14.17% | 17.54% |
| synthetic-00 | 10 | coverage_first | random_global | 240 | 1.9897 | 17.26% | 1.08% |
| synthetic-00 | 10 | coverage_first | random_pool | 240 | 1.7170 | 13.32% | 16.22% |
| synthetic-00 | 20 | blended | blended | 240 | 1.8917 | 20.42% | 20.53% |
| synthetic-00 | 20 | blended | random_global | 240 | 2.2956 | 31.22% | 2.15% |
| synthetic-00 | 20 | blended | random_pool | 240 | 1.9851 | 22.65% | 23.01% |
| synthetic-00 | 20 | coverage_first | coverage_first | 240 | 1.9958 | 22.08% | 25.97% |
| synthetic-00 | 20 | coverage_first | random_global | 240 | 2.2956 | 31.22% | 2.15% |
| synthetic-00 | 20 | coverage_first | random_pool | 240 | 1.9851 | 22.65% | 23.01% |
| synthetic-01 | 5 | blended | blended | 240 | 1.5167 | 7.92% | 12.23% |
| synthetic-01 | 5 | blended | random_global | 240 | 1.6403 | 9.10% | 0.54% |
| synthetic-01 | 5 | blended | random_pool | 240 | 1.4350 | 7.87% | 11.67% |
| synthetic-01 | 5 | coverage_first | coverage_first | 240 | 1.4375 | 6.67% | 12.25% |
| synthetic-01 | 5 | coverage_first | random_global | 240 | 1.6403 | 9.10% | 0.54% |
| synthetic-01 | 5 | coverage_first | random_pool | 240 | 1.4350 | 7.87% | 11.67% |
| synthetic-01 | 10 | blended | blended | 240 | 1.6583 | 12.92% | 16.26% |
| synthetic-01 | 10 | blended | random_global | 240 | 1.9886 | 17.34% | 1.08% |
| synthetic-01 | 10 | blended | random_pool | 240 | 1.7010 | 13.75% | 16.22% |
| synthetic-01 | 10 | coverage_first | coverage_first | 240 | 1.6875 | 11.67% | 17.54% |
| synthetic-01 | 10 | coverage_first | random_global | 240 | 1.9886 | 17.34% | 1.08% |
| synthetic-01 | 10 | coverage_first | random_pool | 240 | 1.7010 | 13.75% | 16.22% |
| synthetic-01 | 20 | blended | blended | 240 | 1.8542 | 18.75% | 21.06% |
| synthetic-01 | 20 | blended | random_global | 240 | 2.2967 | 31.34% | 2.15% |
| synthetic-01 | 20 | blended | random_pool | 240 | 1.9718 | 22.00% | 23.01% |
| synthetic-01 | 20 | coverage_first | coverage_first | 240 | 2.0000 | 26.67% | 25.97% |
| synthetic-01 | 20 | coverage_first | random_global | 240 | 2.2967 | 31.34% | 2.15% |
| synthetic-01 | 20 | coverage_first | random_pool | 240 | 1.9718 | 22.00% | 23.01% |
| synthetic-02 | 5 | blended | blended | 240 | 1.5000 | 8.75% | 12.09% |
| synthetic-02 | 5 | blended | random_global | 240 | 1.6316 | 8.90% | 0.54% |
| synthetic-02 | 5 | blended | random_pool | 240 | 1.4505 | 7.58% | 11.67% |
| synthetic-02 | 5 | coverage_first | coverage_first | 240 | 1.4583 | 8.33% | 12.25% |
| synthetic-02 | 5 | coverage_first | random_global | 240 | 1.6316 | 8.90% | 0.54% |
| synthetic-02 | 5 | coverage_first | random_pool | 240 | 1.4505 | 7.58% | 11.67% |
| synthetic-02 | 10 | blended | blended | 240 | 1.6833 | 13.33% | 15.87% |
| synthetic-02 | 10 | blended | random_global | 240 | 1.9762 | 16.67% | 1.08% |
| synthetic-02 | 10 | blended | random_pool | 240 | 1.7287 | 12.73% | 16.22% |
| synthetic-02 | 10 | coverage_first | coverage_first | 240 | 1.7583 | 12.92% | 17.54% |
| synthetic-02 | 10 | coverage_first | random_global | 240 | 1.9762 | 16.67% | 1.08% |
| synthetic-02 | 10 | coverage_first | random_pool | 240 | 1.7287 | 12.73% | 16.22% |
| synthetic-02 | 20 | blended | blended | 240 | 1.8917 | 18.33% | 20.22% |
| synthetic-02 | 20 | blended | random_global | 240 | 2.2967 | 31.31% | 2.15% |
| synthetic-02 | 20 | blended | random_pool | 240 | 2.0107 | 21.90% | 23.01% |
| synthetic-02 | 20 | coverage_first | coverage_first | 240 | 2.0542 | 24.58% | 25.97% |
| synthetic-02 | 20 | coverage_first | random_global | 240 | 2.2967 | 31.31% | 2.15% |
| synthetic-02 | 20 | coverage_first | random_pool | 240 | 2.0107 | 21.90% | 23.01% |
| synthetic-03 | 5 | blended | blended | 240 | 1.4667 | 8.75% | 12.21% |
| synthetic-03 | 5 | blended | random_global | 240 | 1.6345 | 9.00% | 0.54% |
| synthetic-03 | 5 | blended | random_pool | 240 | 1.4050 | 7.12% | 11.67% |
| synthetic-03 | 5 | coverage_first | coverage_first | 240 | 1.4833 | 10.42% | 12.25% |
| synthetic-03 | 5 | coverage_first | random_global | 240 | 1.6345 | 9.00% | 0.54% |
| synthetic-03 | 5 | coverage_first | random_pool | 240 | 1.4050 | 7.12% | 11.67% |
| synthetic-03 | 10 | blended | blended | 240 | 1.7208 | 16.67% | 16.24% |
| synthetic-03 | 10 | blended | random_global | 240 | 1.9967 | 17.72% | 1.08% |
| synthetic-03 | 10 | blended | random_pool | 240 | 1.6906 | 13.43% | 16.22% |
| synthetic-03 | 10 | coverage_first | coverage_first | 240 | 1.7750 | 16.25% | 17.54% |
| synthetic-03 | 10 | coverage_first | random_global | 240 | 1.9967 | 17.72% | 1.08% |
| synthetic-03 | 10 | coverage_first | random_pool | 240 | 1.6906 | 13.43% | 16.22% |
| synthetic-03 | 20 | blended | blended | 240 | 1.9083 | 21.25% | 20.98% |
| synthetic-03 | 20 | blended | random_global | 240 | 2.2939 | 31.35% | 2.15% |
| synthetic-03 | 20 | blended | random_pool | 240 | 1.9692 | 22.03% | 23.01% |
| synthetic-03 | 20 | coverage_first | coverage_first | 240 | 2.0333 | 27.08% | 25.97% |
| synthetic-03 | 20 | coverage_first | random_global | 240 | 2.2939 | 31.35% | 2.15% |
| synthetic-03 | 20 | coverage_first | random_pool | 240 | 1.9692 | 22.03% | 23.01% |
| synthetic-04 | 5 | blended | blended | 240 | 1.5208 | 6.67% | 12.19% |
| synthetic-04 | 5 | blended | random_global | 240 | 1.6313 | 8.70% | 0.54% |
| synthetic-04 | 5 | blended | random_pool | 240 | 1.4793 | 9.33% | 11.67% |
| synthetic-04 | 5 | coverage_first | coverage_first | 240 | 1.4875 | 5.83% | 12.25% |
| synthetic-04 | 5 | coverage_first | random_global | 240 | 1.6313 | 8.70% | 0.54% |
| synthetic-04 | 5 | coverage_first | random_pool | 240 | 1.4793 | 9.33% | 11.67% |
| synthetic-04 | 10 | blended | blended | 240 | 1.7250 | 14.58% | 16.19% |
| synthetic-04 | 10 | blended | random_global | 240 | 1.9884 | 17.18% | 1.08% |
| synthetic-04 | 10 | blended | random_pool | 240 | 1.7715 | 16.18% | 16.22% |
| synthetic-04 | 10 | coverage_first | coverage_first | 240 | 1.7792 | 14.17% | 17.54% |
| synthetic-04 | 10 | coverage_first | random_global | 240 | 1.9884 | 17.18% | 1.08% |
| synthetic-04 | 10 | coverage_first | random_pool | 240 | 1.7715 | 16.18% | 16.22% |
| synthetic-04 | 20 | blended | blended | 240 | 1.9167 | 22.92% | 21.01% |
| synthetic-04 | 20 | blended | random_global | 240 | 2.3005 | 31.69% | 2.15% |
| synthetic-04 | 20 | blended | random_pool | 240 | 2.0278 | 26.28% | 23.01% |
| synthetic-04 | 20 | coverage_first | coverage_first | 240 | 2.0500 | 25.42% | 25.97% |
| synthetic-04 | 20 | coverage_first | random_global | 240 | 2.3005 | 31.69% | 2.15% |
| synthetic-04 | 20 | coverage_first | random_pool | 240 | 2.0278 | 26.28% | 23.01% |
| synthetic-05 | 5 | blended | blended | 240 | 1.4375 | 7.50% | 12.00% |
| synthetic-05 | 5 | blended | random_global | 240 | 1.6394 | 9.25% | 0.54% |
| synthetic-05 | 5 | blended | random_pool | 240 | 1.4340 | 8.05% | 11.67% |
| synthetic-05 | 5 | coverage_first | coverage_first | 240 | 1.4417 | 7.08% | 12.25% |
| synthetic-05 | 5 | coverage_first | random_global | 240 | 1.6394 | 9.25% | 0.54% |
| synthetic-05 | 5 | coverage_first | random_pool | 240 | 1.4340 | 8.05% | 11.67% |
| synthetic-05 | 10 | blended | blended | 240 | 1.6833 | 12.92% | 15.68% |
| synthetic-05 | 10 | blended | random_global | 240 | 1.9820 | 16.88% | 1.08% |
| synthetic-05 | 10 | blended | random_pool | 240 | 1.7158 | 14.02% | 16.22% |
| synthetic-05 | 10 | coverage_first | coverage_first | 240 | 1.7375 | 13.75% | 17.54% |
| synthetic-05 | 10 | coverage_first | random_global | 240 | 1.9820 | 16.88% | 1.08% |
| synthetic-05 | 10 | coverage_first | random_pool | 240 | 1.7158 | 14.02% | 16.22% |
| synthetic-05 | 20 | blended | blended | 240 | 1.8333 | 17.08% | 19.80% |
| synthetic-05 | 20 | blended | random_global | 240 | 2.2939 | 31.37% | 2.15% |
| synthetic-05 | 20 | blended | random_pool | 240 | 1.9686 | 22.56% | 23.01% |
| synthetic-05 | 20 | coverage_first | coverage_first | 240 | 2.0292 | 24.17% | 25.97% |
| synthetic-05 | 20 | coverage_first | random_global | 240 | 2.2939 | 31.37% | 2.15% |
| synthetic-05 | 20 | coverage_first | random_pool | 240 | 1.9686 | 22.56% | 23.01% |

## Paired comparisons

Positive differences favor the first method. Intervals are paired moving-block bootstrap intervals; Holm adjustment covers all primary comparisons within historical/synthetic families. Short smoke histories are preliminary.

| Job/budget | Comparison | Metric | Difference | 95% interval | Holm p |
|---|---|---|---:|---|---:|
| historical/5 | blended/blended vs blended/random_global | bestHits | -0.1312 | [-0.18173694316436254, -0.09274731182795701] | 1.0000 |
| historical/5 | blended/blended vs blended/random_pool | bestHits | 0.0380 | [-0.00943202764976958, 0.06958563748079878] | 0.3094 |
| historical/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1481 | [-0.19632910906298004, -0.10276267281105994] | 1.0000 |
| historical/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0211 | [-0.016882872503840234, 0.0560526113671275] | 1.0000 |
| historical/5 | coverage_first/coverage_first vs blended/blended | bestHits | -0.0169 | [-0.0445468509984639, 0.021505376344086023] | 1.0000 |
| historical/5 | blended/blended vs blended/random_global | threePlus | 0.0006 | [-0.01797273425499232, 0.019109062980030723] | 1.0000 |
| historical/5 | blended/blended vs blended/random_pool | threePlus | 0.0053 | [-0.012718894009216589, 0.020722350230414743] | 1.0000 |
| historical/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0055 | [-0.022166282642089093, 0.012642473118279561] | 1.0000 |
| historical/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0008 | [-0.015469278033794168, 0.013671658986175107] | 1.0000 |
| historical/5 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0061 | [-0.021505376344086023, 0.010752688172043012] | 1.0000 |
| historical/10 | blended/blended vs blended/random_global | bestHits | -0.2753 | [-0.325516129032258, -0.22784715821812596] | 1.0000 |
| historical/10 | blended/blended vs blended/random_pool | bestHits | -0.0480 | [-0.08310560675883255, -0.011028801843317965] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1846 | [-0.2401232718894009, -0.1300917818740399] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0427 | [0.006236559139784955, 0.08092204301075269] | 0.3720 |
| historical/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0906 | [0.05222734254992319, 0.13060675883256473] | 0.0030 |
| historical/10 | blended/blended vs blended/random_global | threePlus | -0.0336 | [-0.056497695852534564, -0.005790706605222735] | 1.0000 |
| historical/10 | blended/blended vs blended/random_pool | threePlus | -0.0119 | [-0.030215437788018433, 0.007588325652841784] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0059 | [-0.030906682027649766, 0.026221582181259608] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0158 | [-0.0035176651305683523, 0.0391716589861751] | 1.0000 |
| historical/10 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0276 | [0.006144393241167435, 0.05222734254992319] | 0.3650 |
| historical/20 | blended/blended vs blended/random_global | bestHits | -0.4108 | [-0.4625960061443932, -0.37087519201228875] | 1.0000 |
| historical/20 | blended/blended vs blended/random_pool | bestHits | -0.1413 | [-0.18127572964669741, -0.10746390168970817] | 1.0000 |
| historical/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2142 | [-0.2735948540706605, -0.1532246543778802] | 1.0000 |
| historical/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0553 | [0.019215821812596002, 0.0954996159754224] | 0.0837 |
| historical/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1966 | [0.15360983102918588, 0.2488479262672811] | 0.0030 |
| historical/20 | blended/blended vs blended/random_global | threePlus | -0.1203 | [-0.14795698924731182, -0.09568164362519205] | 1.0000 |
| historical/20 | blended/blended vs blended/random_pool | threePlus | -0.0575 | [-0.08265745007680492, -0.037971966205837177] | 1.0000 |
| historical/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0343 | [-0.06576113671274963, 0.006130568356374769] | 1.0000 |
| historical/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0285 | [0.007664362519201229, 0.056590245775729645] | 0.5266 |
| historical/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0860 | [0.05837173579109063, 0.1259600614439324] | 0.0030 |
| synthetic-00/5 | blended/blended vs blended/random_global | bestHits | -0.1827 | [-0.24695937499999998, -0.07903958333333336] | 1.0000 |
| synthetic-00/5 | blended/blended vs blended/random_pool | bestHits | 0.0371 | [-0.029211458333333325, 0.10054374999999999] | 1.0000 |
| synthetic-00/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1993 | [-0.29308333333333336, -0.07129062500000002] | 1.0000 |
| synthetic-00/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0204 | [-0.06450208333333333, 0.10087708333333333] | 1.0000 |
| synthetic-00/5 | coverage_first/coverage_first vs blended/blended | bestHits | -0.0167 | [-0.07916666666666666, 0.0375] | 1.0000 |
| synthetic-00/5 | blended/blended vs blended/random_global | threePlus | -0.0121 | [-0.036375000000000005, 0.02033333333333332] | 1.0000 |
| synthetic-00/5 | blended/blended vs blended/random_pool | threePlus | 0.0010 | [-0.017459375000000003, 0.017791666666666657] | 1.0000 |
| synthetic-00/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | 0.0046 | [-0.032792708333333344, 0.04791770833333332] | 1.0000 |
| synthetic-00/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0177 | [-0.011208333333333336, 0.04175104166666665] | 1.0000 |
| synthetic-00/5 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0167 | [-0.004166666666666667, 0.0375] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_global | bestHits | -0.3147 | [-0.40095937499999995, -0.17891354166666662] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_pool | bestHits | -0.0420 | [-0.12150104166666666, 0.03795937499999999] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2564 | [-0.3324197916666667, -0.12570729166666664] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0163 | [-0.051084375, 0.09333333333333334] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0583 | [0.0125, 0.10833333333333334] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_global | threePlus | -0.0143 | [-0.058709374999999994, 0.04487604166666666] | 1.0000 |
| synthetic-00/10 | blended/blended vs blended/random_pool | threePlus | 0.0251 | [-0.01983333333333333, 0.06604374999999998] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0310 | [-0.07621041666666664, 0.027917708333333315] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0084 | [-0.042503124999999996, 0.05404270833333332] | 1.0000 |
| synthetic-00/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0167 | [-0.05416666666666667, 0.016666666666666666] | 1.0000 |
| synthetic-00/20 | blended/blended vs blended/random_global | bestHits | -0.4040 | [-0.482421875, -0.29133229166666663] | 1.0000 |
| synthetic-00/20 | blended/blended vs blended/random_pool | bestHits | -0.0934 | [-0.16920833333333332, -0.024540625000000017] | 1.0000 |
| synthetic-00/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2998 | [-0.3668395833333333, -0.19058229166666665] | 1.0000 |
| synthetic-00/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0107 | [-0.027959375000000015, 0.056086458333333276] | 1.0000 |
| synthetic-00/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1042 | [0.05, 0.175] | 0.3420 |
| synthetic-00/20 | blended/blended vs blended/random_global | threePlus | -0.1080 | [-0.150209375, -0.05949895833333335] | 1.0000 |
| synthetic-00/20 | blended/blended vs blended/random_pool | threePlus | -0.0223 | [-0.06029374999999999, 0.016417708333333326] | 1.0000 |
| synthetic-00/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0913 | [-0.1375833333333333, -0.03658229166666669] | 1.0000 |
| synthetic-00/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0056 | [-0.03716666666666666, 0.027334374999999987] | 1.0000 |
| synthetic-00/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0167 | [-0.020833333333333332, 0.05416666666666667] | 1.0000 |
| synthetic-01/5 | blended/blended vs blended/random_global | bestHits | -0.1237 | [-0.22679479166666663, -0.016248958333333348] | 1.0000 |
| synthetic-01/5 | blended/blended vs blended/random_pool | bestHits | 0.0817 | [0.02012499999999999, 0.13658437499999998] | 0.3570 |
| synthetic-01/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2028 | [-0.3010427083333333, -0.07133229166666669] | 1.0000 |
| synthetic-01/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0025 | [-0.042667708333333325, 0.06425104166666665] | 1.0000 |
| synthetic-01/5 | coverage_first/coverage_first vs blended/blended | bestHits | -0.0792 | [-0.13333333333333333, -0.008333333333333333] | 1.0000 |
| synthetic-01/5 | blended/blended vs blended/random_global | threePlus | -0.0119 | [-0.05212916666666667, 0.023167708333333318] | 1.0000 |
| synthetic-01/5 | blended/blended vs blended/random_pool | threePlus | 0.0005 | [-0.030459375, 0.022376041666666648] | 1.0000 |
| synthetic-01/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0244 | [-0.051208333333333335, 0.010542708333333307] | 1.0000 |
| synthetic-01/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0120 | [-0.03466770833333333, 0.013374999999999996] | 1.0000 |
| synthetic-01/5 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0125 | [-0.03333333333333333, 0.020833333333333332] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_global | bestHits | -0.3303 | [-0.44187812499999995, -0.19841562499999996] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_pool | bestHits | -0.0427 | [-0.10283333333333333, 0.02437499999999999] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.3011 | [-0.4018822916666666, -0.1789583333333333] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | -0.0135 | [-0.07537604166666666, 0.05904791666666658] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0292 | [-0.025, 0.0875] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_global | threePlus | -0.0442 | [-0.10008437499999999, 0.014127083333333306] | 1.0000 |
| synthetic-01/10 | blended/blended vs blended/random_pool | threePlus | -0.0083 | [-0.057124999999999995, 0.03329166666666667] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0567 | [-0.10079166666666665, -0.00037395833333334153] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0208 | [-0.05408333333333333, 0.020416666666666677] | 1.0000 |
| synthetic-01/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0125 | [-0.041666666666666664, 0.03333333333333333] | 1.0000 |
| synthetic-01/20 | blended/blended vs blended/random_global | bestHits | -0.4425 | [-0.5362083333333333, -0.32591562500000004] | 1.0000 |
| synthetic-01/20 | blended/blended vs blended/random_pool | bestHits | -0.1176 | [-0.16441666666666663, -0.066625] | 1.0000 |
| synthetic-01/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2967 | [-0.38837604166666656, -0.18849895833333333] | 1.0000 |
| synthetic-01/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0282 | [-0.045083333333333336, 0.09329270833333332] | 1.0000 |
| synthetic-01/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1458 | [0.0875, 0.19583333333333333] | 0.0180 |
| synthetic-01/20 | blended/blended vs blended/random_global | threePlus | -0.1259 | [-0.17270937500000003, -0.07725000000000003] | 1.0000 |
| synthetic-01/20 | blended/blended vs blended/random_pool | threePlus | -0.0325 | [-0.07504166666666666, 0.00466666666666667] | 1.0000 |
| synthetic-01/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0468 | [-0.09937604166666668, 0.01862499999999998] | 1.0000 |
| synthetic-01/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0467 | [-0.0018760416666666645, 0.09970833333333334] | 1.0000 |
| synthetic-01/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0792 | [0.04583333333333333, 0.12916666666666668] | 0.0528 |
| synthetic-02/5 | blended/blended vs blended/random_global | bestHits | -0.1316 | [-0.20950104166666667, -0.06820729166666668] | 1.0000 |
| synthetic-02/5 | blended/blended vs blended/random_pool | bestHits | 0.0495 | [-0.0025427083333333243, 0.11600104166666667] | 1.0000 |
| synthetic-02/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1732 | [-0.27404270833333333, -0.106125] | 1.0000 |
| synthetic-02/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0079 | [-0.06033333333333332, 0.06704374999999997] | 1.0000 |
| synthetic-02/5 | coverage_first/coverage_first vs blended/blended | bestHits | -0.0417 | [-0.09166666666666666, -0.008333333333333333] | 1.0000 |
| synthetic-02/5 | blended/blended vs blended/random_global | threePlus | -0.0015 | [-0.04200000000000001, 0.026127083333333294] | 1.0000 |
| synthetic-02/5 | blended/blended vs blended/random_pool | threePlus | 0.0117 | [-0.022252083333333332, 0.04283437499999998] | 1.0000 |
| synthetic-02/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0057 | [-0.05108333333333335, 0.01191666666666666] | 1.0000 |
| synthetic-02/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0076 | [-0.030000000000000002, 0.027958333333333328] | 1.0000 |
| synthetic-02/5 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0042 | [-0.0375, 0.0125] | 1.0000 |
| synthetic-02/10 | blended/blended vs blended/random_global | bestHits | -0.2928 | [-0.39833541666666666, -0.2154083333333334] | 1.0000 |
| synthetic-02/10 | blended/blended vs blended/random_pool | bestHits | -0.0453 | [-0.10883645833333333, 0.0050427083333333235] | 1.0000 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2178 | [-0.3175833333333333, -0.14212395833333333] | 1.0000 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0297 | [-0.029418750000000004, 0.08091770833333332] | 1.0000 |
| synthetic-02/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0750 | [0.029166666666666667, 0.12916666666666668] | 0.6178 |
| synthetic-02/10 | blended/blended vs blended/random_global | threePlus | -0.0333 | [-0.08091770833333334, 0.003291666666666665] | 1.0000 |
| synthetic-02/10 | blended/blended vs blended/random_pool | threePlus | 0.0060 | [-0.029585416666666666, 0.038130208333333256] | 1.0000 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0375 | [-0.08879166666666666, -0.0043333333333333375] | 1.0000 |
| synthetic-02/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0019 | [-0.034541666666666665, 0.02479270833333332] | 1.0000 |
| synthetic-02/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0042 | [-0.0375, 0.025] | 1.0000 |
| synthetic-02/20 | blended/blended vs blended/random_global | bestHits | -0.4050 | [-0.518125, -0.33366354166666673] | 1.0000 |
| synthetic-02/20 | blended/blended vs blended/random_pool | bestHits | -0.1191 | [-0.2023760416666667, -0.05728854166666672] | 1.0000 |
| synthetic-02/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2425 | [-0.33979791666666664, -0.1872041666666667] | 1.0000 |
| synthetic-02/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0434 | [-0.022001041666666697, 0.08254270833333331] | 1.0000 |
| synthetic-02/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1625 | [0.1, 0.225] | 0.0180 |
| synthetic-02/20 | blended/blended vs blended/random_global | threePlus | -0.1298 | [-0.185959375, -0.08670833333333335] | 1.0000 |
| synthetic-02/20 | blended/blended vs blended/random_pool | threePlus | -0.0357 | [-0.07691666666666666, 0.009541666666666667] | 1.0000 |
| synthetic-02/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0673 | [-0.117959375, -0.04562291666666671] | 1.0000 |
| synthetic-02/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0268 | [-0.00575, 0.04758333333333334] | 1.0000 |
| synthetic-02/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0625 | [0.0125, 0.09583333333333334] | 0.1044 |
| synthetic-03/5 | blended/blended vs blended/random_global | bestHits | -0.1679 | [-0.29888125, -0.053872916666666694] | 1.0000 |
| synthetic-03/5 | blended/blended vs blended/random_pool | bestHits | 0.0616 | [-0.003876041666666675, 0.12983333333333333] | 1.0000 |
| synthetic-03/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1512 | [-0.28416875, -0.045615625000000125] | 1.0000 |
| synthetic-03/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0783 | [0.011909374999999998, 0.13045937499999996] | 0.3114 |
| synthetic-03/5 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0167 | [-0.05, 0.075] | 1.0000 |
| synthetic-03/5 | blended/blended vs blended/random_global | threePlus | -0.0025 | [-0.03450104166666667, 0.033917708333333324] | 1.0000 |
| synthetic-03/5 | blended/blended vs blended/random_pool | threePlus | 0.0163 | [-0.008250000000000002, 0.048542708333333316] | 1.0000 |
| synthetic-03/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | 0.0142 | [-0.025583333333333343, 0.052794791666666625] | 1.0000 |
| synthetic-03/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0330 | [0.0038749999999999987, 0.061541666666666654] | 1.0000 |
| synthetic-03/5 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0167 | [-0.008333333333333333, 0.0375] | 1.0000 |
| synthetic-03/10 | blended/blended vs blended/random_global | bestHits | -0.2758 | [-0.40288124999999997, -0.16828541666666674] | 1.0000 |
| synthetic-03/10 | blended/blended vs blended/random_pool | bestHits | 0.0303 | [-0.01995833333333333, 0.08458333333333333] | 1.0000 |
| synthetic-03/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2217 | [-0.36058437499999996, -0.09404062499999996] | 1.0000 |
| synthetic-03/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0844 | [0.01612395833333335, 0.15733437499999997] | 1.0000 |
| synthetic-03/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0542 | [0.0, 0.10833333333333334] | 1.0000 |
| synthetic-03/10 | blended/blended vs blended/random_global | threePlus | -0.0105 | [-0.05379166666666666, 0.03083541666666664] | 1.0000 |
| synthetic-03/10 | blended/blended vs blended/random_pool | threePlus | 0.0323 | [0.005707291666666672, 0.06354166666666666] | 1.0000 |
| synthetic-03/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0147 | [-0.07254374999999999, 0.04416979166666663] | 1.0000 |
| synthetic-03/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0282 | [-0.010084374999999998, 0.0725020833333333] | 1.0000 |
| synthetic-03/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0042 | [-0.04583333333333333, 0.03333333333333333] | 1.0000 |
| synthetic-03/20 | blended/blended vs blended/random_global | bestHits | -0.3856 | [-0.523709375, -0.26099583333333337] | 1.0000 |
| synthetic-03/20 | blended/blended vs blended/random_pool | bestHits | -0.0609 | [-0.11237500000000003, -0.011791666666666688] | 1.0000 |
| synthetic-03/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2606 | [-0.41241770833333324, -0.1283333333333333] | 1.0000 |
| synthetic-03/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0641 | [0.014496874999999985, 0.10958333333333331] | 0.5711 |
| synthetic-03/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1250 | [0.09166666666666666, 0.15833333333333333] | 0.0180 |
| synthetic-03/20 | blended/blended vs blended/random_global | threePlus | -0.1010 | [-0.16445833333333332, -0.03845833333333335] | 1.0000 |
| synthetic-03/20 | blended/blended vs blended/random_pool | threePlus | -0.0078 | [-0.04133541666666666, 0.026541666666666672] | 1.0000 |
| synthetic-03/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0427 | [-0.11158541666666667, 0.03887708333333329] | 1.0000 |
| synthetic-03/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0505 | [0.010916666666666667, 0.0985] | 1.0000 |
| synthetic-03/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0583 | [0.025, 0.1] | 0.3268 |
| synthetic-04/5 | blended/blended vs blended/random_global | bestHits | -0.1104 | [-0.19787812499999996, -0.018621875000000048] | 1.0000 |
| synthetic-04/5 | blended/blended vs blended/random_pool | bestHits | 0.0415 | [0.003165625000000024, 0.08312500000000002] | 1.0000 |
| synthetic-04/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1437 | [-0.23020937499999997, -0.06587395833333333] | 1.0000 |
| synthetic-04/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0082 | [-0.04391770833333335, 0.05654270833333332] | 1.0000 |
| synthetic-04/5 | coverage_first/coverage_first vs blended/blended | bestHits | -0.0333 | [-0.0875, 0.0125] | 1.0000 |
| synthetic-04/5 | blended/blended vs blended/random_global | threePlus | -0.0203 | [-0.04679166666666667, 0.008583333333333328] | 1.0000 |
| synthetic-04/5 | blended/blended vs blended/random_pool | threePlus | -0.0266 | [-0.05129270833333333, -0.0027072916666666864] | 1.0000 |
| synthetic-04/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0286 | [-0.05141770833333334, 0.0032104166666666284] | 1.0000 |
| synthetic-04/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0350 | [-0.051708333333333335, -0.014] | 1.0000 |
| synthetic-04/5 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0083 | [-0.025, 0.016666666666666666] | 1.0000 |
| synthetic-04/10 | blended/blended vs blended/random_global | bestHits | -0.2634 | [-0.3734677083333333, -0.1554927083333334] | 1.0000 |
| synthetic-04/10 | blended/blended vs blended/random_pool | bestHits | -0.0465 | [-0.09345833333333334, -0.009666666666666686] | 1.0000 |
| synthetic-04/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2092 | [-0.3142510416666666, -0.09254062499999996] | 1.0000 |
| synthetic-04/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0076 | [-0.04612604166666669, 0.05858958333333325] | 1.0000 |
| synthetic-04/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0542 | [0.004166666666666667, 0.1125] | 1.0000 |
| synthetic-04/10 | blended/blended vs blended/random_global | threePlus | -0.0260 | [-0.06991666666666667, 0.011918749999999975] | 1.0000 |
| synthetic-04/10 | blended/blended vs blended/random_pool | threePlus | -0.0160 | [-0.0435, 0.004292708333333323] | 1.0000 |
| synthetic-04/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0302 | [-0.07929270833333332, 0.007584374999999992] | 1.0000 |
| synthetic-04/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0201 | [-0.05470833333333333, 0.002960416666666641] | 1.0000 |
| synthetic-04/10 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0042 | [-0.03333333333333333, 0.020833333333333332] | 1.0000 |
| synthetic-04/20 | blended/blended vs blended/random_global | bestHits | -0.3838 | [-0.504084375, -0.2793322916666667] | 1.0000 |
| synthetic-04/20 | blended/blended vs blended/random_pool | bestHits | -0.1111 | [-0.18558437500000002, -0.06387395833333336] | 1.0000 |
| synthetic-04/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2505 | [-0.3410020833333334, -0.15170729166666666] | 1.0000 |
| synthetic-04/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0222 | [-0.03741666666666667, 0.07870833333333331] | 1.0000 |
| synthetic-04/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1333 | [0.075, 0.21666666666666667] | 0.0700 |
| synthetic-04/20 | blended/blended vs blended/random_global | threePlus | -0.0877 | [-0.14483333333333334, -0.039750000000000014] | 1.0000 |
| synthetic-04/20 | blended/blended vs blended/random_pool | threePlus | -0.0336 | [-0.07466666666666667, -0.00433333333333334] | 1.0000 |
| synthetic-04/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0627 | [-0.10900000000000001, -0.010082291666666693] | 1.0000 |
| synthetic-04/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0086 | [-0.04095833333333333, 0.027959374999999984] | 1.0000 |
| synthetic-04/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0250 | [0.0, 0.0625] | 1.0000 |
| synthetic-05/5 | blended/blended vs blended/random_global | bestHits | -0.2019 | [-0.3185864583333333, -0.12695625000000002] | 1.0000 |
| synthetic-05/5 | blended/blended vs blended/random_pool | bestHits | 0.0035 | [-0.053291666666666654, 0.035500000000000004] | 1.0000 |
| synthetic-05/5 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.1977 | [-0.31783333333333336, -0.11929062500000003] | 1.0000 |
| synthetic-05/5 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0077 | [-0.05600104166666667, 0.046001041666666666] | 1.0000 |
| synthetic-05/5 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0042 | [-0.041666666666666664, 0.05010416666666515] | 1.0000 |
| synthetic-05/5 | blended/blended vs blended/random_global | threePlus | -0.0175 | [-0.05770833333333334, 0.01258645833333328] | 1.0000 |
| synthetic-05/5 | blended/blended vs blended/random_pool | threePlus | -0.0055 | [-0.03716979166666667, 0.017791666666666664] | 1.0000 |
| synthetic-05/5 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0216 | [-0.053791666666666675, -0.0023739583333333516] | 1.0000 |
| synthetic-05/5 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0097 | [-0.029709375000000003, 0.00362604166666665] | 1.0000 |
| synthetic-05/5 | coverage_first/coverage_first vs blended/blended | threePlus | -0.0042 | [-0.029166666666666667, 0.025] | 1.0000 |
| synthetic-05/10 | blended/blended vs blended/random_global | bestHits | -0.2987 | [-0.4193749999999999, -0.198121875] | 1.0000 |
| synthetic-05/10 | blended/blended vs blended/random_pool | bestHits | -0.0325 | [-0.08575, 0.016001041666666656] | 1.0000 |
| synthetic-05/10 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2445 | [-0.3662947916666665, -0.14866458333333335] | 1.0000 |
| synthetic-05/10 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0217 | [-0.038126041666666666, 0.06858437499999999] | 1.0000 |
| synthetic-05/10 | coverage_first/coverage_first vs blended/blended | bestHits | 0.0542 | [0.0, 0.09583333333333334] | 1.0000 |
| synthetic-05/10 | blended/blended vs blended/random_global | threePlus | -0.0397 | [-0.07629166666666666, -0.003624999999999995] | 1.0000 |
| synthetic-05/10 | blended/blended vs blended/random_pool | threePlus | -0.0110 | [-0.03849999999999999, 0.02579166666666667] | 1.0000 |
| synthetic-05/10 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0313 | [-0.08292083333333333, 0.005834374999999989] | 1.0000 |
| synthetic-05/10 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | -0.0027 | [-0.042041666666666665, 0.03154270833333332] | 1.0000 |
| synthetic-05/10 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0083 | [-0.025, 0.029166666666666667] | 1.0000 |
| synthetic-05/20 | blended/blended vs blended/random_global | bestHits | -0.4606 | [-0.5882114583333334, -0.3669958333333334] | 1.0000 |
| synthetic-05/20 | blended/blended vs blended/random_pool | bestHits | -0.1353 | [-0.20908437500000004, -0.08637395833333336] | 1.0000 |
| synthetic-05/20 | coverage_first/coverage_first vs coverage_first/random_global | bestHits | -0.2648 | [-0.40549999999999997, -0.16875] | 1.0000 |
| synthetic-05/20 | coverage_first/coverage_first vs coverage_first/random_pool | bestHits | 0.0605 | [-0.022841666666666666, 0.11479374999999996] | 1.0000 |
| synthetic-05/20 | coverage_first/coverage_first vs blended/blended | bestHits | 0.1958 | [0.12916666666666668, 0.2542708333333318] | 0.0180 |
| synthetic-05/20 | blended/blended vs blended/random_global | threePlus | -0.1429 | [-0.20312604166666667, -0.10566562500000003] | 1.0000 |
| synthetic-05/20 | blended/blended vs blended/random_pool | threePlus | -0.0548 | [-0.09808333333333333, -0.02604166666666667] | 1.0000 |
| synthetic-05/20 | coverage_first/coverage_first vs coverage_first/random_global | threePlus | -0.0721 | [-0.14821145833333332, -0.02345729166666669] | 1.0000 |
| synthetic-05/20 | coverage_first/coverage_first vs coverage_first/random_pool | threePlus | 0.0160 | [-0.041460416666666666, 0.05420937499999998] | 1.0000 |
| synthetic-05/20 | coverage_first/coverage_first vs blended/blended | threePlus | 0.0708 | [0.025, 0.11666666666666667] | 0.3718 |

## Runtime and availability

| Job/budget | Objective | Generation ms/target | Model-weighted triple coverage | Mean uncovered triples | Guaranteed targets |
|---|---|---:|---:|---:|---:|
| historical/5 | blended | 35.57 | 0.15041404790157434 | 717.8617511520737 | 0 |
| historical/5 | coverage_first | 23.50 | 0.15383965151609208 | 716.0 | 0 |
| historical/10 | blended | 90.23 | 0.20516063715435356 | 961.7711213517665 | 0 |
| historical/10 | coverage_first | 55.85 | 0.2128448317035623 | 940.0 | 0 |
| historical/20 | blended | 183.74 | 0.27035540974818023 | 1230.857142857143 | 0 |
| historical/20 | coverage_first | 120.52 | 0.30759282210696737 | 1140.0 | 0 |
| synthetic-00/5 | blended | 35.28 | 0.141576985751025 | 716.6083333333333 | 0 |
| synthetic-00/5 | coverage_first | 19.42 | 0.14611753345699852 | 716.0 | 0 |
| synthetic-00/10 | blended | 77.45 | 0.1957895148820776 | 957.0333333333333 | 0 |
| synthetic-00/10 | coverage_first | 46.61 | 0.20435328080352644 | 940.0 | 0 |
| synthetic-00/20 | blended | 171.18 | 0.25932249644547023 | 1223.8416666666667 | 0 |
| synthetic-00/20 | coverage_first | 124.20 | 0.2960737985218396 | 1140.0 | 0 |
| synthetic-01/5 | blended | 38.49 | 0.13759207814392718 | 716.225 | 0 |
| synthetic-01/5 | coverage_first | 25.56 | 0.14220017816989233 | 716.0 | 0 |
| synthetic-01/10 | blended | 84.84 | 0.19029707399469362 | 954.6125 | 0 |
| synthetic-01/10 | coverage_first | 53.66 | 0.19956320645837514 | 940.0 | 0 |
| synthetic-01/20 | blended | 176.30 | 0.2553016989545895 | 1215.6791666666666 | 0 |
| synthetic-01/20 | coverage_first | 119.36 | 0.2899359401223355 | 1140.0 | 0 |
| synthetic-02/5 | blended | 35.69 | 0.14521338888665225 | 717.3333333333334 | 0 |
| synthetic-02/5 | coverage_first | 22.58 | 0.14960792376969423 | 716.0 | 0 |
| synthetic-02/10 | blended | 86.86 | 0.19989101417374558 | 959.0333333333333 | 0 |
| synthetic-02/10 | coverage_first | 56.46 | 0.2086236435700022 | 940.0 | 0 |
| synthetic-02/20 | blended | 187.81 | 0.2642219207251912 | 1228.6083333333333 | 0 |
| synthetic-02/20 | coverage_first | 120.81 | 0.30217291515692196 | 1140.0 | 0 |
| synthetic-03/5 | blended | 35.70 | 0.1378223998245877 | 716.3958333333334 | 0 |
| synthetic-03/5 | coverage_first | 23.25 | 0.1425434121559636 | 716.0 | 0 |
| synthetic-03/10 | blended | 85.46 | 0.1911354282027103 | 954.9208333333333 | 0 |
| synthetic-03/10 | coverage_first | 54.12 | 0.19989779048681816 | 940.0 | 0 |
| synthetic-03/20 | blended | 180.82 | 0.25603554035766346 | 1216.9 | 0 |
| synthetic-03/20 | coverage_first | 123.87 | 0.290488662073076 | 1140.0 | 0 |
| synthetic-04/5 | blended | 35.67 | 0.1407414846001664 | 716.5416666666666 | 0 |
| synthetic-04/5 | coverage_first | 22.53 | 0.1456514046977984 | 716.0 | 0 |
| synthetic-04/10 | blended | 85.75 | 0.19485591807394959 | 955.425 | 0 |
| synthetic-04/10 | coverage_first | 54.76 | 0.20307853641207638 | 940.0 | 0 |
| synthetic-04/20 | blended | 182.82 | 0.26097048365223113 | 1216.4416666666666 | 0 |
| synthetic-04/20 | coverage_first | 118.00 | 0.2940190850719597 | 1140.0 | 0 |
| synthetic-05/5 | blended | 34.70 | 0.14839302429453624 | 718.1125 | 0 |
| synthetic-05/5 | coverage_first | 23.28 | 0.1506480241667207 | 716.0 | 0 |
| synthetic-05/10 | blended | 84.98 | 0.2005712893797139 | 961.2708333333334 | 0 |
| synthetic-05/10 | coverage_first | 56.86 | 0.20839959100778085 | 940.0 | 0 |
| synthetic-05/20 | blended | 177.79 | 0.259013140212672 | 1235.1416666666667 | 0 |
| synthetic-05/20 | coverage_first | 121.09 | 0.3009052635950712 | 1140.0 | 0 |
- historical/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 26692.0749}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 19282.472500000003}]; unavailable forecast counts none.
- historical/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 67064.3535}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 44630.7476}]; unavailable forecast counts none.
- historical/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 135539.2518}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 94834.20499999999}]; unavailable forecast counts none.
- synthetic-00/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 9755.772700000001}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 5848.6392}]; unavailable forecast counts none.
- synthetic-00/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 21086.0453}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 13655.365100000003}]; unavailable forecast counts none.
- synthetic-00/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 46278.3204}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 35529.8826}]; unavailable forecast counts none.
- synthetic-01/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 10602.5981}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 7663.9920999999995}]; unavailable forecast counts none.
- synthetic-01/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 23201.047000000002}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 15785.710300000002}]; unavailable forecast counts none.
- synthetic-01/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 48090.4935}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 34507.5509}]; unavailable forecast counts none.
- synthetic-02/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 9899.8729}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 6920.254900000002}]; unavailable forecast counts none.
- synthetic-02/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 23810.577299999997}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 16530.457300000002}]; unavailable forecast counts none.
- synthetic-02/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 50805.847499999996}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 34546.932700000005}]; unavailable forecast counts none.
- synthetic-03/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 9902.501199999999}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 6988.6446}]; unavailable forecast counts none.
- synthetic-03/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 23433.972999999998}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 15813.059500000003}]; unavailable forecast counts none.
- synthetic-03/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 48951.3318}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 35308.3413}]; unavailable forecast counts none.
- synthetic-04/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 9869.4123}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 6820.045100000001}]; unavailable forecast counts none.
- synthetic-04/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 23421.4928}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 16194.461299999995}]; unavailable forecast counts none.
- synthetic-04/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 49555.5212}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 33867.306099999994}]; unavailable forecast counts none.
- synthetic-05/5: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 5, 'durationMs': 9635.0776}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 5, 'durationMs': 6915.720099999999}]; unavailable forecast counts none.
- synthetic-05/10: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 10, 'durationMs': 23278.5262}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 10, 'durationMs': 16622.417000000005}]; unavailable forecast counts none.
- synthetic-05/20: total evaluation runtime milliseconds [{'objective': 'blended', 'method': 'runtime', 'budget': 20, 'durationMs': 48249.6629}, {'objective': 'coverage_first', 'method': 'runtime', 'budget': 20, 'durationMs': 34737.30069999999}]; unavailable forecast counts none.

## Recommendation

Keep coverage-first research-only: historical coverage, uncertainty or runtime screening is incomplete or did not pass. Synthetic-only results cannot justify UI promotion.

Screening defaults: positive exact coverage gain at every budget; paired lower bounds at least −0.05 best hits and −0.01 three-plus rate; coverage-first generation at most 2 seconds per target. No predictive-superiority claim without independent future evidence.
