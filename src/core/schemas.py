"""Pydantic models as data contracts for the whole system."""

from enum import Enum

from pydantic import BaseModel, Field


class EstimatorType(str, Enum):
    PLR = "plr"
    IRM = "irm"


class LearnerType(str, Enum):
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class SensitivityRequest(BaseModel):
    """Configuration for a sensitivity analysis run."""

    treatment_col: str = Field(default="treatment")
    outcome_col: str = Field(default="outcome")
    estimator: EstimatorType = Field(default=EstimatorType.PLR)
    learners: list[LearnerType] = Field(
        default=[LearnerType.SKLEARN, LearnerType.XGBOOST]
    )
    n_placebo: int = Field(default=100, ge=10, le=1000)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)


# ---------------------------------------------------------------------------
# Result components
# ---------------------------------------------------------------------------

class CausalEstimate(BaseModel):
    estimator: str
    learner: str
    coefficient: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool


class EValueResult(BaseModel):
    evalue_point: float = Field(description="E-value for the point estimate")
    evalue_ci: float = Field(description="E-value for the CI bound closest to null")
    interpretation: str


class PlaceboResult(BaseModel):
    real_ate: float
    placebo_mean: float
    placebo_std: float
    placebo_p_value: float = Field(description="Fraction of permutations with |ATE| >= real |ATE|")
    n_permutations: int
    interpretation: str


class LearnerComparison(BaseModel):
    learner: str
    ate: float
    std_error: float
    ci_lower: float
    ci_upper: float


class SensitivityReport(BaseModel):
    """The full output of a sensitivity analysis."""

    estimation: CausalEstimate
    evalue: EValueResult
    placebo: PlaceboResult
    learner_robustness: list[LearnerComparison]
    dataset_n_obs: int
    dataset_n_features: int


# ---------------------------------------------------------------------------
# Async job tracking (for later extension)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum): # Enum not used yet here, can be used later on with Celery
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"