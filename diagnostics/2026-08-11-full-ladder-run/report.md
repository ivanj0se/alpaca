# Full ladder run: 2026-08-11

## Rung 1: Hawkes branching ratio (trust gate)
- Bar-proxy (SPY): branching_ratio=0.0000 (expect < 0.15) [PASS], 1178 events
- Real-tick replication (SPY): branching_ratio=0.9969 (expect in [0.5, 0.95]) [FAIL], 23470 events

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
| META     |        0.666667 |    0.202167 |   0.231098 |    0.111  |        0.904932 | False         |                8.31017 |
| CAT      |        1        |    0.52     |   0.351923 |    0.2685 |        0.998076 | False         |                5       |
| UNH      |        0.5      |    0.13725  |   0.223131 |    0.2745 |        0.998368 | False         |               26.7772  |
| NVDA     |        0.333333 |    0.199833 |   0.25822  |    0.436  |        0.999989 | False         |                9.09415 |
| GOOGL    |        0.333333 |    0.208833 |   0.232139 |    0.5085 |        0.999999 | False         |                6.6026  |
| AMZN     |        0.666667 |    0.667333 |   0.269567 |    0.742  |        1        | False         |                5       |
| CVX      |        0.5      |    0.6015   |   0.343799 |    0.8445 |        1        | False         |                5       |
| AAPL     |        0        |    0.206333 |   0.23987  |    1      |        1        | False         |               11.476   |
| BA       |        0.5      |    0.68675  |   0.331662 |    0.897  |        1        | False         |                5       |
| WMT      |        0.666667 |    0.928167 |   0.149837 |    0.9835 |        1        | False         |                5       |
| PG       |        0        |    0.20175  |   0.279592 |    1      |        1        | False         |                9.09415 |
| XOM      |        0        |    0.201    |   0.232711 |    1      |        1        | False         |                7.08809 |
| PFE      |        0        |    0.417667 |   0.288866 |    1      |        1        | False         |                5       |
| JNJ      |        0        |    0        |   0        |    1      |        1        | False         |               80.3317  |
| BAC      |        0        |    0.265833 |   0.313846 |    1      |        1        | False         |               22.9519  |
| GS       |        0.5      |    0.8775   |   0.23663  |    0.9805 |        1        | False         |                5       |
| JPM      |        0.5      |    0.797    |   0.282915 |    0.9605 |        1        | False         |                5       |
| MSFT     |        0        |    0.196833 |   0.251619 |    1      |        1        | False         |                6.78859 |
| NEE      |        0        |    0.219833 |   0.236305 |    1      |        1        | False         |                8.16932 |
| LIN      |        0        |    0.1975   |   0.282832 |    1      |        1        | False         |                6.78859 |
