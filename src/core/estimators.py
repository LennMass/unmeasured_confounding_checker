"""DoubleML estimation with optional MLflow tracking."""

import doubleml as dml

from src.core.pipeline import PreparedData
from src.core.learners import get_learners
from src.core.schemas import CausalEstimate


def estimate(
    data: PreparedData,
    estimator: str = "plr",
    learner: str = "sklearn",
    confidence_level: float = 0.95,
    log_mlflow: bool = True,
) -> CausalEstimate:
    """Run a DoubleML estimation and return a CausalEstimate."""

    dml_data = dml.DoubleMLData.from_arrays(x=data.X, y=data.Y, d=data.D)
    ml_l, ml_m = get_learners(learner)

    if estimator == "plr":
        model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m)
    elif estimator == "irm":
        model = dml.DoubleMLIRM(dml_data, ml_g=ml_l, ml_m=ml_m)
    else:
        raise ValueError(f"Unknown estimator '{estimator}'")

    model.fit()
    ci = model.confint(level=confidence_level)

    result = CausalEstimate(
        estimator=estimator.upper(),
        learner=learner,
        coefficient=float(model.coef[0]),
        std_error=float(model.se[0]),
        ci_lower=float(ci.iloc[0, 0]),
        ci_upper=float(ci.iloc[0, 1]),
        p_value=float(model.pval[0]),
        significant=bool(model.pval[0] < (1 - confidence_level)),
    )

    if log_mlflow:
        _log_to_mlflow(result, data)

    return result


def _log_to_mlflow(result: CausalEstimate, data: PreparedData) -> None:
    try:
        import mlflow
        mlflow.set_experiment("unmeasured_confounding_checker")
        with mlflow.start_run():
            mlflow.log_param("estimator", result.estimator)
            mlflow.log_param("learner", result.learner)
            mlflow.log_param("n_obs", data.n_obs)
            mlflow.log_param("n_features", data.n_features)
            mlflow.log_metric("ate", result.coefficient)
            mlflow.log_metric("std_error", result.std_error)
            mlflow.log_metric("p_value", result.p_value)
            mlflow.log_metric("ci_lower", result.ci_lower)
            mlflow.log_metric("ci_upper", result.ci_upper)
    except Exception as e:
        print(f"MLflow logging skipped: {e}")