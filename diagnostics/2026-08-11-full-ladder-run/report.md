# Full ladder run: 2026-08-11

## Rung 1: Hawkes branching ratio (trust gate)
- Bar-proxy (SPY): branching_ratio=0.0000 (expect < 0.15) [PASS], 1178 events
- Real-tick replication: SKIPPED -- 12180 ticks over 0.3 days (need >=5000 over >=5 days)

## Temporal lane (Rung 0 -> Rung 2a GARCH -> Rung 4 TCN-VAE)
| Ticker | Rung0 NLL | Rung2a NLL | 2a beats 0 | Rung4 NLL | 4 beats 2a |
|---|---|---|---|---|---|
| AAPL | -5.7447 | -5.9678 | True | -8.9334 | True |
| AMZN | -5.4836 | -5.8519 | True | -8.1149 | True |
| BA | -5.5267 | -5.6300 | True | -8.5630 | True |
| BAC | -6.0873 | -6.2035 | True | -9.3334 | True |
| CAT | -5.4100 | -5.5357 | True | -8.5435 | True |
| CVX | -5.9861 | -6.0830 | True | -9.1728 | True |
| GOOGL | -5.5721 | -5.8063 | True | -8.6047 | True |
| GS | -5.5315 | -5.6465 | True | -8.5605 | True |
| JNJ | -5.9769 | -6.0992 | True | -9.1713 | True |
| JPM | -6.0325 | -6.1618 | True | -9.2570 | True |
| LIN | -5.9225 | -6.0017 | True | -8.9701 | True |
| META | -5.4078 | -5.7205 | True | -8.3028 | True |
| MSFT | -5.6415 | -5.8962 | True | -8.6200 | True |
| NEE | -6.0820 | -6.1736 | True | -9.3283 | True |
| NVDA | -5.4660 | -5.6080 | True | -8.7203 | True |
| PFE | -6.0269 | -6.1050 | True | -9.2939 | True |
| PG | -5.9997 | -6.1213 | True | -9.1463 | True |
| UNH | -5.8345 | -5.9637 | True | -9.0038 | True |
| WMT | -5.9654 | -6.1010 | True | -9.2130 | True |
| XOM | -5.9076 | -6.0198 | True | -9.1433 | True |

## Cross-sectional lane (Rung 2b factor model -> Rung 3 RPCA)
- Factor model (avg across 20 tickers): mean_nll=-5.6080
- RPCA (whole basket): mean_nll=-6.2063
- RPCA beats factor model: True

## Rung 5: News-correlation attribution
| ticker   |   observed_rate |   null_mean |   null_std |   p_value |   p_value_sidak | significant   |   match_window_minutes |
|:---------|----------------:|------------:|-----------:|----------:|----------------:|:--------------|-----------------------:|
| META     |        0.666667 |    0.204    |   0.229118 |    0.1025 |        0.885002 | False         |                8.31017 |
| CAT      |        1        |    0.53375  |   0.352826 |    0.285  |        0.998781 | False         |                5       |
| UNH      |        0.5      |    0.1445   |   0.226649 |    0.289  |        0.99891  | False         |               26.7772  |
| NVDA     |        0.333333 |    0.197    |   0.257708 |    0.431  |        0.999987 | False         |                9.09415 |
| GOOGL    |        0.333333 |    0.196    |   0.229554 |    0.484  |        0.999998 | False         |                6.6026  |
| AMZN     |        0.666667 |    0.6685   |   0.272125 |    0.7445 |        1        | False         |                5       |
| CVX      |        0.5      |    0.59825  |   0.344923 |    0.841  |        1        | False         |                5       |
| AAPL     |        0        |    0.206833 |   0.244149 |    1      |        1        | False         |               11.476   |
| BA       |        0.5      |    0.6945   |   0.324299 |    0.9085 |        1        | False         |                5       |
| WMT      |        0.666667 |    0.930833 |   0.146607 |    0.9855 |        1        | False         |                5       |
| PG       |        0        |    0.205    |   0.285    |    1      |        1        | False         |                9.09415 |
| XOM      |        0        |    0.203167 |   0.230461 |    1      |        1        | False         |                7.08809 |
| PFE      |        0        |    0.4235   |   0.284279 |    1      |        1        | False         |                5       |
| JNJ      |        0        |    0        |   0        |    1      |        1        | False         |               80.3317  |
| BAC      |        0        |    0.251667 |   0.302026 |    1      |        1        | False         |               22.9519  |
| GS       |        0.5      |    0.8765   |   0.230212 |    0.987  |        1        | False         |                5       |
| JPM      |        0.5      |    0.789    |   0.283688 |    0.961  |        1        | False         |                5       |
| MSFT     |        0        |    0.201333 |   0.252143 |    1      |        1        | False         |                6.78859 |
| NEE      |        0        |    0.221167 |   0.246141 |    1      |        1        | False         |                8.16932 |
| LIN      |        0        |    0.2085   |   0.284654 |    1      |        1        | False         |                6.78859 |
