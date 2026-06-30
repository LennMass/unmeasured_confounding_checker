"""FastAPI application as the HTTP gateway to the sensitivity engine."""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.core.schemas import (
    SensitivityRequest,
    SensitivityReport,
    CausalEstimate,
    EstimatorType,
    LearnerType,
)
from src.core.pipeline import run_pipeline
from src.core.estimators import estimate
from src.core.sensitivity.orchestrator import run_full_analysis

# Avoid OpenMP conflicts between XGBoost and other libs on some platforms
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = FastAPI(
    title="Sensitivity to Unmeasured Confounding",
    description="Robustness Analysis as a Service for ATE estimation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _save_upload(file: UploadFile) -> Path:
    content = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


@app.get("/")
def health():
    return {"status": "ok", "service": "unmeasured_confounding_checker"}


@app.get("/options")
def options():
    return {
        "estimators": [e.value for e in EstimatorType],
        "learners": [l.value for l in LearnerType],
    }


@app.post("/estimate", response_model=CausalEstimate)
async def estimate_endpoint(
    file: UploadFile = File(...),
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    estimator: EstimatorType = EstimatorType.PLR,
    learner: LearnerType = LearnerType.SKLEARN,
    confidence_level: float = 0.95,
):
    """Quick ATE estimate without the full sensitivity suite."""
    path = await _save_upload(file)
    try:
        data = run_pipeline(path, treatment_col, outcome_col)
        result = estimate(
            data,
            estimator=estimator.value,
            learner=learner.value,
            confidence_level=confidence_level,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        path.unlink(missing_ok=True)
    return result


@app.post("/sensitivity", response_model=SensitivityReport)
async def sensitivity_endpoint(
    file: UploadFile = File(...),
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    estimator: EstimatorType = EstimatorType.PLR,
    n_placebo: int = 100,
    confidence_level: float = 0.95,
):
    """Full robustness analysis: ATE + E-value + placebo + learner comparison."""
    path = await _save_upload(file)
    try:
        data = run_pipeline(path, treatment_col, outcome_col)
        request = SensitivityRequest(
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            estimator=estimator,
            learners=[LearnerType.SKLEARN, LearnerType.XGBOOST],
            n_placebo=n_placebo,
            confidence_level=confidence_level,
        )
        report = run_full_analysis(data, request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        path.unlink(missing_ok=True)
    return report