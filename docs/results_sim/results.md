Held-out test: 10 sessions, 5200 frames (split by day). CNN: 27,380 parameters.

| Presence | accuracy | balanced acc. | F1 | AUROC |
|---|---|---|---|---|
| energy | 59.5% | 52.9% | 0.705 | 0.544 |
| logreg | 62.8% | 53.7% | 0.742 | 0.677 |
| cnn | 89.2% | 88.7% | 0.919 | 0.942 |

| Distance (person present) | MAE (m) | RMSE (m) | median (m) | within 25 cm |
|---|---|---|---|---|
| ridge | 0.250 | 0.317 | 0.208 | 58.1% |
| cnn | 0.067 | 0.120 | 0.029 | 95.7% |

| Room ID | frame accuracy | session accuracy (vote) |
|---|---|---|
| logreg | 100.0% | 100.0% |
| cnn | 100.0% | 100.0% |

- presence: CNN balanced accuracy 0.887 vs best baseline (logreg) 0.537 -> CNN better
- distance: CNN MAE 0.067 m vs ridge 0.250 m -> CNN better
- room: CNN frame accuracy 1.000 vs logreg 1.000 -> tie (within noise)
