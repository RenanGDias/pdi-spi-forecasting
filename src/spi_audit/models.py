"""Baselines e modelos tabulares para comparação justa."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def scaled_regressor(regressor):
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", regressor)]),
        transformer=StandardScaler(),
    )


def paper_mlp(hidden_neurons: int = 11, random_state: int = 42):
    """Aproxima a MLP sigmoide/linear e o treino de segunda ordem do artigo."""
    return scaled_regressor(
        MLPRegressor(
            hidden_layer_sizes=(hidden_neurons,),
            activation="logistic",
            solver="lbfgs",
            alpha=0.0,
            max_iter=1000,
            random_state=random_state,
        )
    )


def candidate_regressors(random_state: int = 42) -> dict:
    return {
        "ridge": scaled_regressor(Ridge(alpha=1.0)),
        "mlp_11": paper_mlp(hidden_neurons=11, random_state=random_state),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=120,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
