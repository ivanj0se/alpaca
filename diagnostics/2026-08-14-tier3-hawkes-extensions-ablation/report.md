# Generator comparison: 2026-08-14

## Calibrated band thresholds (per fact, real-data-only)

| Fact | Threshold | Alpha |
|---|---|---|
| raw_return_acf | 0.02763 | 0.10 |
| volatility_clustering_acf | 0.11580 | 0.10 |
| excess_kurtosis | 12.10955 | 0.10 |
| leverage_curve | 0.03706 | 0.10 |
| aggregational_kurtosis | 7.51605 | 0.10 |

## Ranking (higher overall_score is better)

| generator_id | overall_score | n_paths | coverage_raw_return_acf | coverage_volatility_clustering_acf | coverage_excess_kurtosis | coverage_leverage_curve | coverage_aggregational_kurtosis |
|---|---|---|---|---|---|---|---|
| hawkes_multi_kernel | 0.9040 | 25 | 0.8800 | 0.8400 | 1.0000 | 0.8400 | 0.9600 |
| hawkes_treatment | 0.8640 | 25 | 0.7200 | 0.9600 | 1.0000 | 0.6400 | 1.0000 |
| cox_hawkes_rpca | 0.5200 | 25 | 0.6800 | 0.1200 | 0.6400 | 0.6000 | 0.5600 |
| hawkes_control | 0.4000 | 25 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

## Per-fact mean distances

| generator_id | raw_return_acf | volatility_clustering_acf | excess_kurtosis | leverage_curve | aggregational_kurtosis |
|---|---|---|---|---|---|
| hawkes_control | 0.01874 | 0.20949 | 13.38861 | 0.02313 | 8.73138 |
| hawkes_treatment | 0.02688 | 0.05908 | 7.19377 | 0.03859 | 4.32459 |
| hawkes_multi_kernel | 0.02497 | 0.09296 | 6.18719 | 0.03105 | 4.13017 |
| cox_hawkes_rpca | 0.02875 | 0.18440 | 14.40170 | 0.03871 | 9.25362 |