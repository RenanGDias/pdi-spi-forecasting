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


def paper_mlp(
    hidden_neurons: int = 11,
    random_state: int = 42,
    early_stopping: bool = False,
):
    """Aproxima a MLP sigmoide/linear e o treino de segunda ordem do artigo.

    O artigo declara que o treino é interrompido quando o MSE da validação
    deixa de cair, mas o L-BFGS de segunda ordem não expõe essa parada. Com
    ``early_stopping=True`` a otimização passa a ser iterativa e reserva uma
    fração do treino para a parada, o que permite testar se as conclusões da
    auditoria dependem do ajuste sem parada.
    """
    if early_stopping:
        return scaled_regressor(
            MLPRegressor(
                hidden_layer_sizes=(hidden_neurons,),
                activation="logistic",
                solver="adam",
                alpha=0.0,
                max_iter=2000,
                early_stopping=True,
                # 15/85: reproduz a proporção validação/treino declarada pelo artigo.
                validation_fraction=0.1765,
                n_iter_no_change=15,
                random_state=random_state,
            )
        )
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
