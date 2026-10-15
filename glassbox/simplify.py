"""Making a trained rule base shorter without making it less faithful.

Two greedy steps:

1. Condition pruning - repeatedly delete the single condition ("AND x is T")
   whose removal hurts fidelity least.
2. Rule pruning - then delete whole rules the same way.

An edit is only accepted if BOTH
  - fidelity to the network (validation pool) stays within `tolerance` of the
    starting value, and
  - the squared error between fuzzy risk and network probability (training
    pool) grows by at most `mse_growth` (25 %).
The second check matters: fidelity only counts flipped decisions, so on its
own it lets pruning push many patients right up to the 0.5 threshold, and
those flip on new data. (Without it, test fidelity fell from 94 % to 92 %.)

Rules that become identical after pruning are merged. After every edit the
rule outputs (THEN-parts) are re-fitted by least squares.
"""
import numpy as np

from . import config
from .metrics import fidelity
from .rulebase import DONT_CARE, unique_rules


def _score(model, A, X_train, t_train, X_val, nn_val_labels):
    candidate = model.with_antecedents(A, X_train, t_train)
    fid = fidelity(candidate.predict(X_val), nn_val_labels)
    mse = float(np.mean((candidate.predict_proba(X_train) - t_train) ** 2))
    return fid, mse, candidate


def _greedy(model, edits, X_train, t_train, X_val, nn_val_labels, floor, mse_limit):
    """Apply the best acceptable edit until no edit is acceptable."""
    log = []
    while True:
        best = None
        for description, A in edits(model):
            fid, mse, candidate = _score(model, A, X_train, t_train, X_val, nn_val_labels)
            if fid >= floor and mse <= mse_limit and (best is None or (fid, -mse) > (best[0], -best[1])):
                best = (fid, mse, candidate, description)
        if best is None:
            return model, log
        model = best[2]
        log.append(f"{best[3]} -> fidelity {best[0]:.3f}")


def _condition_removals(model):
    A = model.antecedents_
    for r, j in zip(*np.nonzero(A != DONT_CARE)):
        B = A.copy()
        B[r, j] = DONT_CARE
        term = model.term_names[A[r, j]]
        yield f"drop '{model.feature_names[j]} is {term}' from rule {r + 1}", unique_rules(B)


def _rule_removals(model):
    A = model.antecedents_
    if len(A) <= 1:
        return
    for r in range(len(A)):
        yield f"drop rule {r + 1}", np.delete(A, r, axis=0)


def simplify(model, X_train, t_train, X_val, nn_val_labels, tolerance=config.SIMPLIFY_TOLERANCE,
             mse_growth=config.SIMPLIFY_MSE_GROWTH):
    """Return (simplified model, list of edits made)."""
    t_train = np.asarray(t_train, dtype=float)
    base = fidelity(model.predict(X_val), nn_val_labels)
    floor = base - tolerance
    mse_limit = float(np.mean((model.predict_proba(X_train) - t_train) ** 2)) * (1 + mse_growth)
    args = (X_train, t_train, X_val, nn_val_labels, floor, mse_limit)
    model, log1 = _greedy(model, _condition_removals, *args)
    model, log2 = _greedy(model, _rule_removals, *args)
    return model, [f"start: fidelity {base:.3f}"] + log1 + log2
