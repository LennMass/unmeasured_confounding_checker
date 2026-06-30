"""Orchestrator: run the full sensitivity analysis end to end."""

from src.core.pipeline import PreparedData
from src.core.estimators import estimate
from src.core.sensitivity.evalue import compute_evalue
from src.core.sensitivity.placebo import run_placebo_test
from src.core.sensitivity.learner_compare import compare_learners
from src.core.schemas import SensitivityReport, SensitivityRequest


def run_full_analysis(data: PreparedData, request: SensitivityRequest) -> SensitivityReport:
    """Run estimation + E-value + placebo + learner robustness."""

    learners = [l.value for l in request.learners]
    primary_learner = learners[0]

    # 1. Primary estimation
    primary = estimate(
        data,
        estimator=request.estimator.value,
        learner=primary_learner,
        confidence_level=request.confidence_level,
    )

    # 2. E-value
    evalue = compute_evalue(primary)

    # 3. Placebo test
    placebo = run_placebo_test(
        data,
        primary,
        estimator=request.estimator.value,
        learner=primary_learner,
        n_permutations=request.n_placebo,
    )

    # 4. Learner robustness
    learner_robustness = compare_learners(
        data,
        learners=learners,
        estimator=request.estimator.value,
        confidence_level=request.confidence_level,
    )

    return SensitivityReport(
        estimation=primary,
        evalue=evalue,
        placebo=placebo,
        learner_robustness=learner_robustness,
        dataset_n_obs=data.n_obs,
        dataset_n_features=data.n_features,
    )