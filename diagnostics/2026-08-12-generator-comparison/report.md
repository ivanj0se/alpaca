# Generator comparison: 2026-08-12

## Calibrated band thresholds (per fact, real-data-only)

| Fact | Threshold | Alpha |
|---|---|---|
| raw_return_acf | 0.00850 | 0.10 |
| volatility_clustering_acf | 0.07254 | 0.10 |
| excess_kurtosis | 4.95632 | 0.10 |
| leverage_curve | 0.01141 | 0.10 |
| aggregational_kurtosis | 3.26948 | 0.10 |

## Ranking (higher overall_score is better)

| generator_id | overall_score | n_paths | coverage_raw_return_acf | coverage_volatility_clustering_acf | coverage_excess_kurtosis | coverage_leverage_curve | coverage_aggregational_kurtosis |
|---|---|---|---|---|---|---|---|
| hawkes_treatment | 0.1680 | 25 | 0.0000 | 0.6800 | 0.0400 | 0.0000 | 0.1200 |
| gbm_null | 0.0000 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hawkes_control | 0.0000 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tcn_forecaster | 0.0000 | 25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per-fact mean distances

| generator_id | raw_return_acf | volatility_clustering_acf | excess_kurtosis | leverage_curve | aggregational_kurtosis |
|---|---|---|---|---|---|
| gbm_null | 0.01919 | 0.20929 | 14.69064 | 0.02346 | 9.12164 |
| hawkes_control | 0.01834 | 0.21045 | 13.36892 | 0.02337 | 8.72662 |
| hawkes_treatment | 0.02725 | 0.06180 | 7.96079 | 0.02937 | 4.67383 |
| tcn_forecaster | 0.01947 | 0.20903 | 14.71535 | 0.02435 | 9.07682 |