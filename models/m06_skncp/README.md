# model6 — SKNCP Task1

Task1(click prediction F1)을 위해 SKNCP(Semantic k-Nearest Click Prediction)를
중심 feature로 쓰는 독립 실험 디렉터리다.

기본 SKNCP sweep:

```bash
python -X utf8 models/m06_skncp/experiments/task1_skncp_model6.py
```

F1 0.11+ boosted SKNCP:

```bash
python -X utf8 models/m04_gated/experiments/exp_content_strong.py
python -X utf8 models/m06_skncp/experiments/task1_skncp_boosted.py
```

결과:

- `models/m06_skncp/results/model6_task1_results.json`
- `models/m06_skncp/results/model6_boosted_task1_results.json`
- `models/m06_skncp/MODEL6_SKNCP_TASK1_REPORT.md`

현재 boosted validation 결과: F1 `0.1109`, AUC `0.7130`.
