# Full ladder run: 2026-08-11

## Rung 1: Hawkes branching ratio (trust gate)
- Bar-proxy (SPY): branching_ratio=0.0000 (expect < 0.15) [PASS], 760 events
- Real-tick replication: SKIPPED -- no real tick data recorded yet

## Temporal lane (Rung 0 -> Rung 2a GARCH -> Rung 4 TCN-VAE)
| Ticker | Rung0 NLL | Rung2a NLL | 2a beats 0 | Rung4 NLL | 4 beats 2a |
|---|---|---|---|---|---|
| AAPL | -5.6090 | -5.8532 | True | -5.6136 | False |
| AMZN | -5.2722 | -5.6584 | True | -5.2717 | False |
| BA | -5.2457 | -5.3731 | True | -5.2480 | False |
| BAC | -5.8354 | -6.0403 | True | -5.8385 | False |
| CAT | -4.8617 | -4.9781 | True | -4.8607 | False |
| CVX | -5.5691 | -5.7285 | True | -5.5712 | False |
| GOOGL | -5.3754 | -5.6517 | True | -5.3791 | False |
| GS | -5.1468 | -5.1999 | True | -5.1499 | False |
| JNJ | -5.6345 | -5.8554 | True | -5.6385 | False |
| JPM | -5.7152 | -5.8832 | True | -5.7147 | False |
| LIN | -5.3876 | -5.5939 | True | -5.3816 | False |
| META | -5.1425 | -5.4730 | True | -5.1439 | False |
| MSFT | -5.2678 | -5.6726 | True | -5.2687 | False |
| NEE | -5.8826 | -6.0495 | True | -5.8882 | False |
| NVDA | -5.2488 | -5.4165 | True | -5.2502 | False |
| PFE | -5.7818 | -5.9419 | True | -5.7865 | False |
| PG | -5.5912 | -5.7998 | True | -5.5926 | False |
| UNH | -5.3217 | -5.4174 | True | -5.3211 | False |
| WMT | -5.5896 | -5.8067 | True | -5.5945 | False |
| XOM | -5.4883 | -5.6768 | True | -5.4912 | False |

## Cross-sectional lane (Rung 2b factor model -> Rung 3 RPCA)
- Factor model (avg across 20 tickers): mean_nll=-5.4580
- RPCA (whole basket): mean_nll=-6.0271
- RPCA beats factor model: True

## Rung 5: News-correlation attribution
| ticker   |   observed_rate |   null_mean |   null_std |   p_value |   p_value_sidak | significant   |
|:---------|----------------:|------------:|-----------:|----------:|----------------:|:--------------|
| UNH      |        0.666667 |    0.185333 |  0.207863  |    0.071  |        0.770747 | False         |
| NVDA     |        1        |    0.425833 |  0.341808  |    0.107  |        0.896001 | False         |
| META     |        0.666667 |    0.553833 |  0.337625  |    0.5665 |        1        | False         |
| XOM      |        0.666667 |    0.625167 |  0.320416  |    0.666  |        1        | False         |
| GOOGL    |        0.666667 |    0.640833 |  0.309389  |    0.6725 |        1        | False         |
| NEE      |        0.333333 |    0.5965   |  0.35374   |    0.8415 |        1        | False         |
| BA       |        1        |    0.99875  |  0.0249687 |    0.9975 |        1        | False         |
| CAT      |        0.5      |    0.9875   |  0.0780625 |    1      |        1        | False         |
| WMT      |        1        |    1        |  0         |    1      |        1        | False         |
| PG       |        0        |    0        |  0         |    1      |        1        | False         |
| AMZN     |        0        |    0        |  0         |    1      |        1        | False         |
| AAPL     |        0.333333 |    0.4545   |  0.276761  |    0.852  |        1        | False         |
| PFE      |        0        |    0.964667 |  0.132398  |    1      |        1        | False         |
| JNJ      |        0        |    0        |  0         |    1      |        1        | False         |
| BAC      |        0        |    0        |  0         |    1      |        1        | False         |
| GS       |        1        |    1        |  0         |    1      |        1        | False         |
| JPM      |        1        |    0.999833 |  0.0074517 |    0.9995 |        1        | False         |
| MSFT     |        0        |    0.626333 |  0.382878  |    1      |        1        | False         |
| CVX      |        1        |    0.9985   |  0.027345  |    0.997  |        1        | False         |
| LIN      |        0        |    0        |  0         |    1      |        1        | False         |
