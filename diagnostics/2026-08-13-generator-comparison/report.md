# Generator comparison: 2026-08-13

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
| hawkes_treatment | 0.8960 | 25 | 0.7200 | 0.9200 | 1.0000 | 0.8400 | 1.0000 |
| gbm_null | 0.4000 | 25 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| zero_intelligence | 0.4000 | 25 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| hawkes_control | 0.4000 | 25 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| tcn_forecaster | 0.4000 | 25 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

## Per-fact mean distances

| generator_id | raw_return_acf | volatility_clustering_acf | excess_kurtosis | leverage_curve | aggregational_kurtosis |
|---|---|---|---|---|---|
| gbm_null | 0.01919 | 0.20929 | 14.69064 | 0.02346 | 9.12164 |
| zero_intelligence | 0.01900 | 0.20965 | 14.66672 | 0.02553 | 8.99640 |
| hawkes_control | 0.01962 | 0.20993 | 13.31327 | 0.02353 | 8.58039 |
| hawkes_treatment | 0.02706 | 0.06166 | 7.69574 | 0.02944 | 4.83095 |
| tcn_forecaster | 0.02143 | 0.20940 | 14.71572 | 0.02392 | 9.06931 |