"""End-to-end pipeline: data -> black-box network -> fuzzy explainer.

Used by both `scripts/run_experiments.py` and the Streamlit demo so that they
always build exactly the same models.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from . import config
from .anfis import ANFIS
from .blackbox import BlackBox, make_mlp, tune_mlp
from .data import Dataset, load_dataset
from .feature_selection import permutation_importance, top_features
from .oracle import oracle_augment
from .simplify import simplify


@dataclass
class FuzzySurrogate:
    selected: list                 # column indices of the features given to ANFIS
    full: ANFIS                    # after training, before simplification
    model: ANFIS                   # after simplification (the one we present)
    simplify_log: list = field(default_factory=list)


@dataclass
class GlassBox:
    data: Dataset
    blackbox: BlackBox
    nn_params: dict
    importance: pd.DataFrame
    surrogate: FuzzySurrogate

    @property
    def selected_names(self):
        return [self.data.feature_names[j] for j in self.surrogate.selected]


def train_blackbox(data: Dataset, params=None, seed=config.SEED):
    """Tune with cross-validation unless known-good `params` are supplied."""
    if params is None:
        pipeline, params, _ = tune_mlp(data.X_train, data.y_train, seed=seed)
    else:
        params = {"hidden_layer_sizes": tuple(params["hidden_layer_sizes"]), "alpha": params["alpha"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline = make_mlp(seed=seed, **params).fit(data.X_train, data.y_train)
    return BlackBox(pipeline), params


def fit_fuzzy_surrogate(
    data: Dataset,
    blackbox: BlackBox,
    selected,
    n_terms=config.N_TERMS,
    n_rules=config.N_RULES,
    rule_init="fcm",
    target="nn",
    oracle_samples=config.ORACLE_SAMPLES,
    constrain=True,
    do_simplify=True,
    epochs=config.ANFIS_EPOCHS,
    fine_tune_epochs=config.FINE_TUNE_EPOCHS,
    order=0,
    seed=config.SEED,
) -> FuzzySurrogate:
    """Train ANFIS to imitate the network (target="nn") or the true labels (target="truth").

    "truth" is a control experiment: it shows the difference between
    *explaining the network* and simply *solving the task again*.
    """
    selected = list(selected)
    names = [data.feature_names[j] for j in selected]
    if target == "nn":
        X_fit = data.X_train
        t_fit = blackbox.predict_proba(data.X_train)
        if oracle_samples:
            X_syn, t_syn = oracle_augment(data.X_train, blackbox.predict_proba, oracle_samples, seed=seed)
            X_fit, t_fit = np.vstack([X_fit, X_syn]), np.concatenate([t_fit, t_syn])
        # The validation pool is enlarged the same way, so that early stopping and
        # pruning decisions are not swayed by one or two of the 154 real patients.
        X_val = data.X_val
        if oracle_samples:
            X_syn, _ = oracle_augment(data.X_val, blackbox.predict_proba, oracle_samples // 2, seed=seed + 1)
            X_val = np.vstack([X_val, X_syn])
        t_val = blackbox.predict_proba(X_val)
        val_labels = (t_val >= blackbox.threshold).astype(int)
    elif target == "truth":
        X_fit, t_fit = data.X_train, data.y_train.astype(float)
        X_val, t_val = data.X_val, data.y_val.astype(float)
        val_labels = data.y_val
    else:
        raise ValueError("target must be 'nn' or 'truth'")

    X_fit, X_val = X_fit[:, selected], X_val[:, selected]
    full = ANFIS(names, n_terms=n_terms, n_rules=n_rules, rule_init=rule_init,
                 constrain=constrain, epochs=epochs, order=order, seed=seed).fit(X_fit, t_fit, X_val, t_val)
    if not do_simplify:
        return FuzzySurrogate(selected, full, full, [])
    model, log = simplify(full, X_fit, t_fit, X_val, val_labels)
    model = model.fine_tune(X_fit, t_fit, X_val, t_val, epochs=fine_tune_epochs)
    return FuzzySurrogate(selected, full, model, log)


def train_glassbox(seed=config.SEED, nn_params=None, n_features=config.N_FEATURES, **surrogate_kwargs) -> GlassBox:
    data = load_dataset(seed=seed)
    blackbox, nn_params = train_blackbox(data, nn_params, seed=seed)
    importance = permutation_importance(blackbox.predict_proba, data.X_val, data.feature_names, seed=seed)
    selected = top_features(importance, data.feature_names, n_features)
    surrogate = fit_fuzzy_surrogate(data, blackbox, selected, seed=seed, **surrogate_kwargs)
    return GlassBox(data, blackbox, nn_params, importance, surrogate)
