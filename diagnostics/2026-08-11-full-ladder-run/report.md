# Full ladder run: 2026-08-11

## Rung 1: Hawkes branching ratio (trust gate)
- Bar-proxy (SPY): branching_ratio=0.0000 (expect < 0.15) [PASS], 760 events
- Real-tick replication: SKIPPED -- 1694 ticks over 0.1 days (need >=5000 over >=5 days)

## Temporal lane (Rung 0 -> Rung 2a GARCH -> Rung 4 TCN-VAE)
| Ticker | Rung0 NLL | Rung2a NLL | 2a beats 0 | Rung4 NLL | 4 beats 2a |
|---|---|---|---|---|---|
| AAPL | -5.6090 | -5.8532 | True | -8.8112 | True |
| AMZN | -5.2722 | -5.6584 | True | -8.2919 | True |
| BA | -5.2457 | -5.3731 | True | -8.3450 | True |
| BAC | -5.8354 | -6.0403 | True | -9.0590 | True |
| CAT | -4.8617 | -4.9781 | True | -7.8255 | True |
| CVX | -5.5691 | -5.7285 | True | -8.7314 | True |
| GOOGL | -5.3754 | -5.6517 | True | -8.5773 | True |
| GS | -5.1468 | -5.1999 | True | -8.1712 | True |
| JNJ | -5.6345 | -5.8554 | True | -8.7169 | True |
| JPM | -5.7152 | -5.8832 | True | -8.8695 | True |
| LIN | -5.3876 | -5.5939 | True | -8.0521 | True |
| META | -5.1425 | -5.4730 | True | -8.1627 | True |
| MSFT | -5.2678 | -5.6726 | True | -8.2781 | True |
| NEE | -5.8826 | -6.0495 | True | -9.1180 | True |
| NVDA | -5.2488 | -5.4165 | True | -8.4966 | True |
| PFE | -5.7818 | -5.9419 | True | -9.0320 | True |
| PG | -5.5912 | -5.7998 | True | -8.6642 | True |
| UNH | -5.3217 | -5.4174 | True | -8.2804 | True |
| WMT | -5.5896 | -5.8067 | True | -8.7183 | True |
| XOM | -5.4883 | -5.6768 | True | -8.6486 | True |

## Cross-sectional lane (Rung 2b factor model -> Rung 3 RPCA)
- Factor model (avg across 20 tickers): mean_nll=-5.4580
- RPCA (whole basket): mean_nll=-6.0271
- RPCA beats factor model: True

## Rung 5: News-correlation attribution
| ticker   |   observed_rate |   null_mean |   null_std |   p_value |   p_value_sidak | significant   |
|:---------|----------------:|------------:|-----------:|----------:|----------------:|:--------------|
| BAC      |        1        |    0.327    |  0.344906  |    0.0965 |        0.86861  | False         |
| AAPL     |        1        |    0.450667 |  0.405668  |    0.259  |        0.997509 | False         |
| UNH      |        0.666667 |    0.283667 |  0.316368  |    0.266  |        0.99794  | False         |
| META     |        1        |    0.557    |  0.423839  |    0.4025 |        0.999966 | False         |
| NVDA     |        0.666667 |    0.537833 |  0.287154  |    0.556  |        1        | False         |
| NEE      |        0.666667 |    0.592167 |  0.324474  |    0.6425 |        1        | False         |
| GOOGL    |        0        |    0.648    |  0.41871   |    1      |        1        | False         |
| BA       |        1        |    0.9995   |  0.0158035 |    0.999  |        1        | False         |
| CAT      |        1        |    0.98625  |  0.0996792 |    0.979  |        1        | False         |
| WMT      |        1        |    1        |  0         |    1      |        1        | False         |
| PG       |        0        |    0.5315   |  0.456079  |    1      |        1        | False         |
| XOM      |        0        |    0.6205   |  0.281922  |    1      |        1        | False         |
| CVX      |        1        |    0.99625  |  0.0431386 |    0.9925 |        1        | False         |
| PFE      |        0.333333 |    0.964167 |  0.123353  |    1      |        1        | False         |
| JNJ      |        0        |    0.107    |  0.205063  |    1      |        1        | False         |
| GS       |        1        |    1        |  0         |    1      |        1        | False         |
| JPM      |        1        |    0.999667 |  0.0105357 |    0.999  |        1        | False         |
| MSFT     |        0.333333 |    0.633    |  0.274428  |    0.952  |        1        | False         |
| AMZN     |        1        |    0.999167 |  0.0166458 |    0.9975 |        1        | False         |
| LIN      |        0.5      |    0.6305   |  0.346366  |    0.8565 |        1        | False         |
