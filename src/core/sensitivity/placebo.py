"""
Placebo (permutation) test: shuffle the treatment assignment many times and
re-estimate the effect. If the real effect is genuine, the permuted effects
should cluster around zero and the real estimate should sit far in the tail.
"""

import numpy as np

from src.core.pipeline import PreparedData
from src.core.estimators import estimate
from src.core.schemas import PlaceboResult, CausalEstimate


def run_placebo_test(
    data: PreparedData,
    real_estimate: CausalEstimate,
    estimator: str = "plr",
    learner: str = "sklearn",
    n_permutations: int = 100,
    seed: int = 42,
) -> PlaceboResult:
    """Permute the treatment vector and collect placebo ATEs."""

    rng = np.random.default_rng(seed)
    placebo_ates = []

    for _ in range(n_permutations):
        permuted_D = rng.permutation(data.D)
        placebo_data = PreparedData(
            X=data.X,
            D=permuted_D,
            Y=data.Y,
            feature_names=data.feature_names,
            n_obs=data.n_obs,
            n_features=data.n_features,
        )
        try:
            est = estimate(
                placebo_data,
                estimator=estimator,
                learner=learner,
                log_mlflow=False,
            )
            placebo_ates.append(est.coefficient)
        except Exception:
            continue

    placebo_ates = np.array(placebo_ates)
    real_abs = abs(real_estimate.coefficient)
    p_value = float(np.mean(np.abs(placebo_ates) >= real_abs)) if len(placebo_ates) else 1.0

    interpretation = (
        f"Across {len(placebo_ates)} permutations of the treatment, the placebo "
        f"effects average {placebo_ates.mean():.3f} (std {placebo_ates.std():.3f}). "
        f"The real effect of {real_estimate.coefficient:.3f} sits at a placebo "
        f"p-value of {p_value:.3f}, meaning {p_value*100:.1f}% of random treatment "
        f"assignments produced an effect at least as large in magnitude."
    )

    return PlaceboResult(
        real_ate=round(real_estimate.coefficient, 4),
        placebo_mean=round(float(placebo_ates.mean()), 4) if len(placebo_ates) else 0.0,
        placebo_std=round(float(placebo_ates.std()), 4) if len(placebo_ates) else 0.0,
        placebo_p_value=round(p_value, 4),
        n_permutations=len(placebo_ates),
        interpretation=interpretation,
    )