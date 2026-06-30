# Sensitivity to Unmeasured Confounding in ATE estimation

**Robustness Analysis as a Service**

Estimate an average treatment effect with DoubleML, then automatically probe
how robust that estimate is: how strong an unmeasured confounder would need
to be to explain it away (E-value), whether the effect survives placebo
permutations, and whether it holds across different ML nuisance learners.

A minimal, adaptable starting point with FastAPI backend, Streamlit dashboard,
MLflow tracking, Docker, and CI.

---

## What it does

Upload observational data with a treatment column, an outcome column, and
confounders. You get back:

- **Estimation** — the ATE with confidence interval and p-value (DoubleML)
- **E-value** — minimum strength an unmeasured confounder would need to nullify the effect
- **Placebo test** — re-estimate under permuted treatment; the real effect should sit far in the tail
- **Learner robustness** — the same effect estimated with multiple ML backends (sklearn, XGBoost)

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│  Sensitivity │
│  dashboard  │◀────│   gateway   │◀────│    engine    │
└─────────────┘     └─────────────┘     └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │    MLflow    │
                                        │   tracking   │
                                        └──────────────┘
```

The sensitivity engine is plain Python (`src/core/`), so you can use it as a
library, behind the API, or from notebooks — independent of the web layer.

---

## Quickstart (local)

```bash
# 1. Install
pip install -e ".[all]"

# 2. Start the API
make api          # uvicorn on :8000

# 3. In another terminal, start the dashboard
make dashboard    # streamlit on :8501

# 4. (optional) MLflow UI
make mlflow       # on :5001
```

Open the dashboard at http://localhost:8501, upload `data/sample_data.csv`,
and click **Run Analysis**.

---

## Quickstart (Docker)

```bash
make docker-up
```

Brings up API (`:8000`), dashboard (`:8501`), and MLflow (`:5001`) together.

---

## Example output

Using the bundled `data/sample_data.csv` (800 rows, true treatment effect = 300):

```
ATE:            274.23   (p < 0.001)
E-value:        robust — a confounder would need a very large association
                to explain the effect away
Placebo test:   p = 0.00 — no permuted treatment reproduced the effect
Learner check:  sklearn = 280.7,  xgboost = 271.4   (stable)
```

Both learners agree, the placebo distribution is centered near zero, and the
real effect sits in the extreme tail — all signs of a robust result.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/options` | GET | Available estimators and learners |
| `/estimate` | POST | Quick ATE estimate |
| `/sensitivity` | POST | Full robustness analysis |

Interactive docs at http://localhost:8000/docs once the API is running.

Example:

```bash
curl -X POST "http://localhost:8000/sensitivity?treatment_col=treatment&outcome_col=outcome&n_placebo=100" \
  -F "file=@data/sample_data.csv"
```

---

## Project structure

```
causal-sensitivity/
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI app
│   └── core/
│       ├── pipeline.py             # Polars data prep
│       ├── learners.py             # sklearn / XGBoost factory
│       ├── estimators.py           # DoubleML + MLflow logging
│       ├── schemas.py              # Pydantic contracts
│       └── sensitivity/
│           ├── evalue.py           # E-value
│           ├── placebo.py          # permutation test
│           ├── learner_compare.py  # multi-learner robustness
│           └── orchestrator.py     # runs the full suite
├── dashboard/
│   └── app.py                      # Streamlit UI
├── tests/                          # pytest suite
├── data/
│   └── sample_data.csv             # toy dataset, true effect = 300
├── docker/
│   ├── Dockerfile                  # multi-stage (api / dashboard)
│   └── docker-compose.yml          # api + dashboard + mlflow
├── .github/workflows/test.yml      # CI
├── Makefile
└── pyproject.toml
```

---

## Adapting it to your needs

This is deliberately minimal. Common extensions:

- **More sensitivity methods** — add a module under `src/core/sensitivity/`
  (e.g. Partial R² / Cinelli–Hazlett, Rosenbaum bounds) and wire it into
  `orchestrator.py` and the `SensitivityReport` schema.
- **More learners** — add a factory in `learners.py` (LightGBM, TabPFN, a
  neural net) and register it in `get_learners()`.
- **Async jobs** — for large datasets, move `/sensitivity` onto a Celery
  worker and return a job ID.
- **Kubernetes** — the Docker images are deployment-ready; add manifests
  under a `k8s/` directory.

---

## Testing

```bash
make test
```

Runs the full suite: pipeline, estimation, E-value formula, placebo test,
learner comparison, and all API endpoints.

---

## License

MIT
