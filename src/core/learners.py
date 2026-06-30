"""Nuisance learner factory for DoubleML."""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def get_sklearn_learners() -> tuple:
    ml_l = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=100, random_state=42)
    return ml_l, ml_m


def get_xgboost_learners() -> tuple:
    from xgboost import XGBRegressor, XGBClassifier
    ml_l = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
    ml_m = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0, eval_metric="logloss")
    return ml_l, ml_m


def get_learners(name: str = "sklearn") -> tuple:
    factories = {
        "sklearn": get_sklearn_learners,
        "xgboost": get_xgboost_learners,
    }
    if name not in factories:
        raise ValueError(f"Unknown learner '{name}'. Choose from: {list(factories.keys())}")
    return factories[name]()