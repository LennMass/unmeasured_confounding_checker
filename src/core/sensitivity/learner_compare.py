"""
Multi-learner robustness: estimate the same effect with several nuisance
learners. If the estimate is stable across learners, that's evidence the
result is not an artifact of one particular ML model.
"""

from src.core.pipeline import PreparedData
from src.core.estimators import estimate
from src.core.schemas import LearnerComparison


def compare_learners(
    data: PreparedData,
    learners: list[str],
    estimator: str = "plr",
    confidence_level: float = 0.95,
) -> list[LearnerComparison]:
    """Run the same estimator with each learner and collect the results."""

    results = []
    for learner in learners:
        try:
            est = estimate(
                data,
                estimator=estimator,
                learner=learner,
                confidence_level=confidence_level,
                log_mlflow=False,
            )
            results.append(LearnerComparison(
                learner=learner,
                ate=round(est.coefficient, 4),
                std_error=round(est.std_error, 4),
                ci_lower=round(est.ci_lower, 4),
                ci_upper=round(est.ci_upper, 4),
            ))
        except Exception as e:
            print(f"Learner {learner} failed: {e}")

    return results