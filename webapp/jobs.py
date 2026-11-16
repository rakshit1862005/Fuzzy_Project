"""Training runs for the live lab.

A run executes the whole pipeline in a background thread with the settings
chosen in the browser, and records an *event* at every step (one per network
epoch, per ANFIS epoch, per pruning edit, ...). The browser receives the
events as they happen (server-sent events) and draws them live.

Nothing here is precomputed: every run trains a new network and a new fuzzy
system from the raw CSV.
"""
import threading
import time
import traceback
import uuid
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score

from glassbox import config
from glassbox.blackbox import BlackBox, make_mlp
from glassbox.data import load_dataset
from glassbox.explain import WEAK_MATCH, explain_instance
from glassbox.feature_selection import permutation_importance, top_features
from glassbox.metrics import anfis_complexity, classification_metrics, fidelity, fidelity_by_confidence
from glassbox.pipeline import fit_fuzzy_surrogate
from glassbox.rulebase import DONT_CARE, unique_rules
from glassbox.surrogates import fit_tree_surrogate, tree_complexity

HIDDEN_LAYERS = {"8": (8,), "16": (16,), "16,8": (16, 8), "32,16": (32, 16)}

# The settings used in the report; the browser starts with these.
DEFAULTS = {
    "hidden_layers": "8",
    "alpha": 1.0,
    "tune": False,
    "n_features": config.N_FEATURES,
    "n_terms": config.N_TERMS,
    "n_rules": config.N_RULES,
    "order": 0,
    "epochs": config.ANFIS_EPOCHS,
    "oracle_samples": config.ORACLE_SAMPLES,
    "constrain": True,
    "target": "nn",
    "prune": True,
    "tolerance": config.SIMPLIFY_TOLERANCE,
    "seed": config.SEED,
    "slow_motion": True,
}

# Pauses (seconds) used in slow motion, so a ~5 s run can be followed by eye.
SLOW = {"nn_epoch": 0.008, "anfis_epoch": 0.012, "prune_step": 0.35, "stage": 0.4}

OPTIONS = {
    "hidden_layers": list(HIDDEN_LAYERS),
    "alpha": [0.001, 0.01, 0.1, 1.0],
    "n_features": [2, 6],
    "n_terms": [2, 3],
    "n_rules": [2, 16],
    "order": [0, 1],
    "epochs": [20, 600],
    "oracle_samples": [0, 4000],
    "target": ["nn", "truth"],
    "tolerance": [0.0, 0.05],
}


def clean_config(raw: dict) -> dict:
    """Merge browser settings with the defaults, coercing types and clamping ranges."""
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    if cfg["hidden_layers"] not in HIDDEN_LAYERS:
        raise ValueError(f"hidden_layers must be one of {list(HIDDEN_LAYERS)}")
    if cfg["target"] not in ("nn", "truth"):
        raise ValueError("target must be 'nn' or 'truth'")

    def clamp(key, cast):
        lo, hi = OPTIONS[key]
        cfg[key] = min(max(cast(cfg[key]), lo), hi)

    for key in ("n_features", "n_rules", "epochs", "oracle_samples"):
        clamp(key, int)
    clamp("tolerance", float)
    cfg["alpha"] = float(cfg["alpha"])
    cfg["n_terms"] = 2 if int(cfg["n_terms"]) == 2 else 3
    cfg["order"] = 1 if int(cfg["order"]) == 1 else 0
    cfg["seed"] = int(cfg["seed"])
    for key in ("tune", "constrain", "prune", "slow_motion"):
        cfg[key] = bool(cfg[key])
    return cfg


def to_py(value):
    """NumPy values -> plain Python, so events can be sent as JSON."""
    if isinstance(value, dict):
        return {str(k): to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_py(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_py(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ------------------------------------------------------------ model -> JSON
def describe_model(model) -> dict:
    """Everything the browser needs to draw the terms and rules itself."""
    features = []
    for j, name in enumerate(model.feature_names):
        features.append({
            "name": name,
            "unit": config.UNITS.get(name, ""),
            "lo": model.lo_[j],
            "hi": model.hi_[j],
            "terms": [{"name": term, "center": model.centers_original[j, t], "sigma": model.sigmas_original[j, t]}
                      for t, term in enumerate(model.term_names)],
        })
    rules = []
    for r, antecedent in enumerate(model.antecedents_):
        rule = {"id": r + 1, "antecedent": antecedent.tolist(),
                "conditions": [{"feature": j, "term": int(t)} for j, t in enumerate(antecedent) if t != DONT_CARE]}
        if model.order == 0:
            rule["output"] = float(model.consequents_[r])
        else:
            from glassbox.rules import linear_then_part
            rule["formula"] = linear_then_part(model, r)
        rules.append(rule)
    return to_py({"features": features, "term_names": model.term_names, "rules": rules,
                  "order": model.order, "threshold": model.threshold})


# --------------------------------------------------------------------- runs
class Job:
    def __init__(self, cfg: dict):
        self.id = uuid.uuid4().hex[:8]
        self.cfg = cfg
        self.created = time.time()
        self.status = "running"
        self.events = []
        self._cond = threading.Condition()
        self.summary = None          # final metrics, shown in the runs table
        self.data = self.blackbox = self.surrogate = None
        self.model = None            # the rule base being explored / edited

    # events ------------------------------------------------------------
    def emit(self, kind: str, **payload):
        if kind == "stage" and payload.get("status") == "done":
            self.pace("stage")
        with self._cond:
            self.events.append(to_py({"id": len(self.events), "type": kind, "t": time.time() - self.created,
                                      **payload}))
            self._cond.notify_all()

    def pace(self, what: str):
        """In slow motion, pause briefly so the browser animation can be followed."""
        if self.cfg.get("slow_motion"):
            time.sleep(SLOW[what])

    def wait_events(self, start: int, timeout: float = 15.0) -> list:
        with self._cond:
            if start >= len(self.events) and self.status == "running":
                self._cond.wait(timeout)
            return self.events[start:]

    @property
    def finished(self):
        return self.status != "running"

    # evaluation --------------------------------------------------------
    def evaluate(self, model) -> dict:
        """Test-set fidelity, accuracy and size of a rule base."""
        sel = self.surrogate.selected
        Xt = self.data.X_test[:, sel]
        nn_prob = self.blackbox.predict_proba(self.data.X_test)
        nn_lab = (nn_prob >= self.blackbox.threshold).astype(int)
        pred = model.predict(Xt)
        split = fidelity_by_confidence(pred, nn_lab, nn_prob)
        match = model.firing_strengths(Xt)[0].max(axis=1)
        return to_py({
            **anfis_complexity(model),
            "fidelity": fidelity(pred, nn_lab),
            "fidelity_boundary": split["boundary"],
            "fidelity_confident": split["confident"],
            "accuracy": classification_metrics(self.data.y_test, model.predict_proba(Xt))["accuracy"],
            "weak_match_share": float(np.mean(match < WEAK_MATCH)),
            "disagreements": int(np.sum(pred != nn_lab)),
        })


class JobStore:
    """In-memory list of runs (the most recent `keep` are kept)."""

    def __init__(self, keep: int = 25):
        self.keep = keep
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, raw_cfg: dict) -> Job:
        job = Job(clean_config(raw_cfg))
        with self._lock:
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-self.keep]:
                del self._jobs[old.id]
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job:
        return self._jobs.get(job_id)

    def all(self) -> list:
        return sorted(self._jobs.values(), key=lambda j: j.created)


# ------------------------------------------------------------- the pipeline
def run_job(job: Job):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = job.cfg
    try:
        _run(job, cfg)
        job.status = "done"
    except Exception as exc:  # report any failure to the browser instead of dying silently
        job.status = "failed"
        job.emit("error", message=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
    finally:
        with job._cond:
            job._cond.notify_all()


def _run(job: Job, cfg: dict):
    # 1. data ---------------------------------------------------------------
    job.emit("stage", stage="data", status="running")
    data = load_dataset(seed=config.SEED)
    job.data = data
    all_x = np.vstack([data.X_train, data.X_val, data.X_test])
    job.emit("data", train=len(data.y_train), val=len(data.y_val), test=len(data.y_test),
             diabetic_share=float(np.mean(np.r_[data.y_train, data.y_val, data.y_test])),
             medians=data.medians, features=data.feature_names)
    job.emit("stage", stage="data", status="done")

    # 2. neural network -----------------------------------------------------
    job.emit("stage", stage="network", status="running")
    hidden, alpha = HIDDEN_LAYERS[cfg["hidden_layers"]], cfg["alpha"]
    if cfg["tune"]:
        grid = list(ParameterGrid(config.NN_PARAM_GRID))        # same order as GridSearchCV
        folds = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
        scores = []
        for i, params in enumerate(grid):
            auc = cross_val_score(make_mlp(params["mlp__hidden_layer_sizes"], params["mlp__alpha"]),
                                  data.X_train, data.y_train, cv=folds, scoring="roc_auc")
            scores.append(auc.mean())
            job.emit("cv", index=i, total=len(grid), hidden=list(params["mlp__hidden_layer_sizes"]),
                     alpha=params["mlp__alpha"], auc=float(auc.mean()), std=float(auc.std()))
        best = grid[int(np.argmax(scores))]
        hidden, alpha = best["mlp__hidden_layer_sizes"], best["mlp__alpha"]
        job.emit("cv_done", hidden=list(hidden), alpha=alpha)

    pipeline = make_mlp(hidden, alpha)
    mlp = pipeline.named_steps["mlp"]
    scaler = pipeline.named_steps["scale"]
    X_val_scaled = []

    def on_nn_epoch(model, epoch):
        if not X_val_scaled:                       # the scaler is fitted before the first epoch
            X_val_scaled.append(scaler.transform(data.X_val))
        val_acc = float(np.mean(model.predict(X_val_scaled[0]) == data.y_val))
        job.emit("nn_epoch", epoch=epoch, loss=float(model.loss_), val_accuracy=val_acc)
        job.pace("nn_epoch")

    mlp.on_epoch = on_nn_epoch
    pipeline.fit(data.X_train, data.y_train)
    mlp.on_epoch = None
    bb = BlackBox(pipeline)
    job.blackbox = bb
    nn_test = classification_metrics(data.y_test, bb.predict_proba(data.X_test))
    job.emit("nn_done", description=bb.describe(), hidden=list(hidden), alpha=alpha, test=nn_test)
    job.emit("stage", stage="network", status="done")

    # 3. what the network relies on ------------------------------------------
    job.emit("stage", stage="importance", status="running")
    importance = permutation_importance(bb.predict_proba, data.X_val, data.feature_names)
    selected = top_features(importance, data.feature_names, cfg["n_features"])
    job.emit("importance", rows=importance.to_dict(orient="records"),
             selected=[data.feature_names[j] for j in selected])
    job.emit("stage", stage="importance", status="done")

    # 4-6. fuzzy explainer: FCM rules -> ANFIS -> pruning -> fine-tuning -------
    state = {"stage": None}

    def on_progress(stage, model, **info):
        if stage == "anfis" and state["stage"] is None:
            job.emit("stage", stage="rules", status="running")
            fcm = getattr(model, "fcm_", None)
            job.emit("rules_proposed", objective=fcm.objective if fcm is not None else [],
                     clusters=cfg["n_rules"], model=describe_model(model))
            job.emit("stage", stage="rules", status="done")
            job.emit("stage", stage="anfis", status="running")
        if stage != state["stage"]:
            if state["stage"] == "anfis":
                job.emit("stage", stage="anfis", status="done")
                job.emit("stage", stage="prune", status="running")
            state["stage"] = stage
        if stage in ("anfis", "fine_tune"):
            event = {"phase": stage, "epoch": info["epoch"], "train_loss": info["train_loss"],
                     "val_loss": info["val_loss"]}
            if info["epoch"] % 5 == 0 or info["epoch"] == 1:
                event["model"] = describe_model(model)
            job.emit("anfis_epoch", **event)
            job.pace("anfis_epoch")
        elif stage == "prune":
            job.emit("prune_step", description=info["description"], fidelity=info["fidelity"],
                     **{k: v for k, v in anfis_complexity(model).items() if k in ("rules", "conditions")},
                     model=describe_model(model))
            job.pace("prune_step")

    surrogate = fit_fuzzy_surrogate(
        data, bb, selected, n_terms=cfg["n_terms"], n_rules=cfg["n_rules"], target=cfg["target"],
        oracle_samples=cfg["oracle_samples"] if cfg["target"] == "nn" else 0, constrain=cfg["constrain"],
        do_simplify=cfg["prune"], epochs=cfg["epochs"], order=cfg["order"], tolerance=cfg["tolerance"],
        seed=cfg["seed"], on_progress=on_progress)
    job.surrogate = surrogate
    job.model = surrogate.model
    if state["stage"] == "anfis":                  # pruning switched off: ANFIS was the last step
        job.emit("stage", stage="anfis", status="done")
    job.emit("stage", stage="prune", status="done" if cfg["prune"] else "skipped")

    # 7. evaluation -----------------------------------------------------------
    job.emit("stage", stage="evaluate", status="running")
    full, final = job.evaluate(surrogate.full), job.evaluate(surrogate.model)
    sel_names = [data.feature_names[j] for j in selected]
    tree = fit_tree_surrogate(data.X_train[:, selected], bb.predict(data.X_train), final["rules"])
    tree_fid = fidelity(tree.predict(data.X_test[:, selected]), bb.predict(data.X_test))
    job.summary = {"full": full, "final": final, "nn_accuracy": nn_test["accuracy"],
                   "tree": {**tree_complexity(tree), "fidelity": tree_fid}}
    job.emit("done",
             summary=job.summary,
             model=describe_model(surrogate.model),
             selected=sel_names,
             selected_index=selected,
             feature_names=data.feature_names,
             units=[config.UNITS.get(f, "") for f in data.feature_names],
             ranges={"lo": all_x.min(axis=0), "hi": all_x.max(axis=0)},
             typical=np.median(data.X_train, axis=0),
             patients={"x": data.X_test, "y": data.y_test,
                       "nn_prob": bb.predict_proba(data.X_test)},
             log=surrogate.simplify_log)
    job.emit("stage", stage="evaluate", status="done")


# ------------------------------------------------------------ interaction
def predict(job: Job, x) -> dict:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape[0] != len(job.data.feature_names):
        raise ValueError(f"expected {len(job.data.feature_names)} values")
    nn_prob = float(job.blackbox.predict_proba(x[None])[0])
    return to_py(explain_instance(job.model, x[job.surrogate.selected], nn_prob=nn_prob))


def edit_rules(job: Job, antecedents) -> dict:
    """Replace the rule base by an edited one; THEN-parts are re-fitted by least squares."""
    base = job.surrogate.model
    A = np.asarray(antecedents, dtype=int)
    if A.ndim != 2 or A.shape[1] != base.antecedents_.shape[1] or len(A) == 0:
        raise ValueError("antecedents must be a non-empty list of rules, one entry per input")
    if np.any((A < DONT_CARE) | (A >= base.n_terms)):
        raise ValueError(f"terms must be -1 (not used) or 0..{base.n_terms - 1}")
    job.model = base.with_antecedents(unique_rules(A), job.surrogate.X_fit, job.surrogate.t_fit)
    return {"model": describe_model(job.model), "metrics": job.evaluate(job.model),
            "trained": job.summary["final"]}


def run_row(job: Job) -> dict:
    row = {"id": job.id, "status": job.status, "created": job.created, "config": job.cfg}
    if job.summary:
        row["summary"] = job.summary
        if job.model is not None and job.model is not job.surrogate.model:
            row["edited"] = job.evaluate(job.model)
    return to_py(row)
