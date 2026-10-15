"""The black-box model: a multilayer perceptron trained with backpropagation.

The rest of the project never looks inside this network. It only asks it
"what is P(diabetic) for this patient?" - which is exactly the situation of
someone who has to explain a model they cannot open.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def make_mlp(hidden_layer_sizes=(16, 8), alpha=1e-3, seed=config.SEED) -> Pipeline:
    """Standardise inputs, then a ReLU MLP trained with Adam for a fixed number of epochs.

    `alpha` is the L2 weight penalty that controls overfitting. We tried
    early stopping first, but with only ~460 training patients its held-out
    slice (~70 patients) was too small and made results jump between runs.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    return Pipeline([("scale", StandardScaler()), ("mlp", mlp)])


def tune_mlp(X, y, seed=config.SEED, param_grid=None, folds=5):
    """Grid search over architecture and regularisation, scored by ROC-AUC.

    Returns (fitted best pipeline, best params, cv_results_ dict).
    """
    search = GridSearchCV(
        make_mlp(seed=seed),
        param_grid or config.NN_PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X, y)
    params = {
        "hidden_layer_sizes": search.best_params_["mlp__hidden_layer_sizes"],
        "alpha": search.best_params_["mlp__alpha"],
    }
    return search.best_estimator_, params, search.cv_results_


class BlackBox:
    """Wrapper exposing the network as an oracle: inputs in, P(diabetic) out."""

    def __init__(self, pipeline: Pipeline, threshold: float = config.DECISION_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def describe(self) -> str:
        mlp = self.pipeline.named_steps["mlp"]
        layers = [mlp.n_features_in_, *mlp.hidden_layer_sizes, 1]
        n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
        return (
            f"MLP {' -> '.join(map(str, layers))} (ReLU, sigmoid output), "
            f"{n_params} trainable parameters, L2 alpha={mlp.alpha}, "
            f"trained for {mlp.n_iter_} epochs"
        )
