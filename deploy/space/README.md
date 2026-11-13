---
title: Cell EOL Forecaster
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Li-ion cell end-of-life forecasts with abstain-to-engineer
---

# Li-ion cell end-of-life forecaster

Forecasts the cycle at which an 18650 cell reaches 80 % of its initial capacity
from its early cycles (NASA PCoE aging data), with a 90 % interval. Inputs
outside the training envelope (other temperatures, discharge currents, glitches,
bad records) are routed to an engineer or rejected instead of answered.

- Demo page with 22 built-in test cases: `/`
- API docs: `/docs`. Forecast: `POST /v1/forecast`. Cases: `GET /v1/cases`, `POST /v1/cases/{id}/run`
- Source, training pipeline (MLflow + Kubeflow) and evaluation:
  https://github.com/ZaifMohammed-eng/nasa-cell-lifetime-model
