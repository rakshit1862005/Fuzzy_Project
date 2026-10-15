"""ANFIS: Adaptive Neuro-Fuzzy Inference System (Jang, 1993), written with NumPy.

We use a zero-order Takagi-Sugeno system, so every rule has the form

    Rule r:  IF x_1 is A_r1 AND ... AND x_k is A_rk  THEN  y = p_r

where p_r is a single number (the diabetes risk this rule predicts). This is
far easier to read than first-order rules with a linear formula after THEN.

The five ANFIS layers, for one input vector x:

    1. Fuzzification     mu_jt(x_j) = exp(-(x_j - c_jt)^2 / (2 s_jt^2))
                         ("Low" and "High" are shoulders: flat at 1 beyond their centre)
    2. Rule firing       w_r  = product of the memberships named in rule r  (AND = product)
    3. Normalisation     wn_r = w_r / sum_q w_q
    4. Consequents       wn_r * p_r
    5. Output            y = sum_r wn_r * p_r          (weighted average of rule outputs)

Hybrid learning (one epoch):
    forward pass  - premises (c, s) fixed, so y is *linear* in p:
                    solve for p with (ridge-regularised) least squares.
    backward pass - p fixed, update c and s by gradient descent on the
                    mean squared error (we use the Adam optimiser).

Interpretability constraints after every update keep the terms meaningful:
each centre may drift at most `max_shift` from its starting position (so
"Low" stays low), centres stay sorted (Low < Medium < High) and neighbours
keep at least half of their original distance, and each width stays between half and double its starting value (so no
term shrinks to a spike or swallows its neighbours).

All computations happen on inputs min-max scaled to [0, 1]; centres and widths
are converted back to original units only for display.
"""
import copy

import numpy as np

from . import config
from .membership import initial_partition, log_memberships, term_names, term_offsets
from .rulebase import DONT_CARE, fcm_rules, grid_rules


class Adam:
    """The Adam optimiser (Kingma & Ba, 2015) for a list of NumPy arrays."""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * g ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ANFIS:
    def __init__(
        self,
        feature_names,
        n_terms=config.N_TERMS,
        n_rules=config.N_RULES,
        rule_init="fcm",
        antecedents=None,
        epochs=config.ANFIS_EPOCHS,
        lr=config.ANFIS_LR,
        ridge=1.0,
        constrain=True,
        min_gap_ratio=0.5,
        max_shift=0.1,
        sigma_range=(0.5, 2.0),
        threshold=config.DECISION_THRESHOLD,
        seed=config.SEED,
    ):
        self.feature_names = list(feature_names)
        self.n_terms = n_terms
        self.term_names = term_names(n_terms)
        self.n_rules = n_rules
        self.rule_init = rule_init
        self.antecedents = antecedents
        self.epochs = epochs
        self.lr = lr
        self.ridge = ridge
        self.constrain = constrain
        self.min_gap_ratio = min_gap_ratio
        self.max_shift = max_shift
        self.sigma_range = sigma_range
        self.threshold = threshold
        self.seed = seed

    # ------------------------------------------------------------ scaling
    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.lo_) / self.span_

    @property
    def centers_original(self):
        return self.lo_[:, None] + self.centers_ * self.span_[:, None]

    @property
    def sigmas_original(self):
        return self.sigmas_ * self.span_[:, None]

    # ------------------------------------------------------- layers 1 - 3
    def _log_memberships(self, Xn, centers=None, sigmas=None):
        c = self.centers_ if centers is None else centers
        s = self.sigmas_ if sigmas is None else sigmas
        return log_memberships(Xn, c, s)                                # (n, k, m)

    def _log_firing(self, Xn, centers=None, sigmas=None, antecedents=None):
        """log w_r = sum of log-memberships in rule r (log of the product).

        Working with logs avoids underflow when many small memberships are
        multiplied; it is the same maths.
        """
        A = self.antecedents_ if antecedents is None else antecedents
        log_mu = self._log_memberships(Xn, centers, sigmas)
        log_w = np.zeros((len(Xn), len(A)))
        for j in range(A.shape[1]):
            used = A[:, j] != DONT_CARE
            log_w[:, used] += log_mu[:, j, A[used, j]]
        return log_w

    @staticmethod
    def _normalize(log_w):
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)

    # ----------------------------------------------------- hybrid learning
    def _solve_consequents(self, wbar, target):
        """Least squares for p: minimise ||wbar p - t||^2 + ridge * ||p - prior||^2.

        The small ridge term acts like one imaginary patient of average risk
        per rule, so a rule that almost never fires gets a neutral output
        instead of an arbitrary one. Outputs are clipped to [0, 1] because
        they are probabilities.
        """
        R = wbar.shape[1]
        lhs = wbar.T @ wbar + self.ridge * np.eye(R)
        rhs = wbar.T @ target + self.ridge * self.prior_
        return np.clip(np.linalg.solve(lhs, rhs), 0.0, 1.0)

    def _premise_gradients(self, Xn, wbar, y, target):
        """dLoss/dc and dLoss/ds for the loss L = mean((y - t)^2), with p held fixed.

        Chain rule: dL/dtheta = sum_n sum_r G[n, r] * dlog(w_r)/dtheta, where
            G[n, r] = 2/N * (y_n - t_n) * wn_r * (p_r - y_n)
        and for a Gaussian term used by rule r, with d = x - c:
            dlog(mu)/dc = d / s^2,     dlog(mu)/ds = d^2 / s^3
        (on the flat part of a shoulder d = 0, so the gradient is 0 there).
        """
        p = self.consequents_
        G = (2.0 / len(Xn)) * (y - target)[:, None] * wbar * (p[None, :] - y[:, None])
        grad_c = np.zeros_like(self.centers_)
        grad_s = np.zeros_like(self.sigmas_)
        offsets = term_offsets(Xn, self.centers_)
        for j in range(self.centers_.shape[0]):
            for t in range(self.n_terms):
                rules_using = self.antecedents_[:, j] == t
                if not rules_using.any():
                    continue
                g = G[:, rules_using].sum(axis=1)
                diff = offsets[:, j, t]
                s = self.sigmas_[j, t]
                grad_c[j, t] = np.sum(g * diff) / s ** 2
                grad_s[j, t] = np.sum(g * diff ** 2) / s ** 3
        return grad_c, grad_s

    def _apply_constraints(self):
        if not self.constrain:
            self.sigmas_ = np.maximum(self.sigmas_, 1e-3)
            return
        self.centers_ = np.clip(self.centers_,                # limited drift, inside the data range
                                np.maximum(self.initial_centers_ - self.max_shift, 0.0),
                                np.minimum(self.initial_centers_ + self.max_shift, 1.0))
        order = np.argsort(self.centers_, axis=1)            # keep Low < Medium < High
        self.centers_ = np.take_along_axis(self.centers_, order, axis=1)
        self.sigmas_ = np.take_along_axis(self.sigmas_, order, axis=1)
        min_gaps = self.min_gap_ratio * np.diff(self.initial_centers_, axis=1)
        for t in range(1, self.n_terms):                     # terms may not merge
            self.centers_[:, t] = np.maximum(self.centers_[:, t], self.centers_[:, t - 1] + min_gaps[:, t - 1])
        self.sigmas_ = np.clip(self.sigmas_, self.initial_sigmas_ * self.sigma_range[0],
                               self.initial_sigmas_ * self.sigma_range[1])

    def fit(self, X, target, X_val=None, target_val=None):
        """Train on inputs X and targets in [0, 1] (here: the network's probabilities).

        If validation data is given, the premise parameters from the epoch
        with the lowest validation error are kept (early stopping).
        """
        X = np.asarray(X, dtype=float)
        target = np.asarray(target, dtype=float)
        self.lo_ = X.min(axis=0)
        self.hi_ = X.max(axis=0)
        self.span_ = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        self.prior_ = float(target.mean())
        Xn = self._scale(X)

        parts = [initial_partition(Xn[:, j], self.n_terms) for j in range(X.shape[1])]
        self.centers_ = np.array([c for c, _ in parts])
        self.sigmas_ = np.array([s for _, s in parts])
        self.initial_centers_ = self.centers_.copy()
        self.initial_sigmas_ = self.sigmas_.copy()

        if self.antecedents is not None:
            self.antecedents_ = np.asarray(self.antecedents, dtype=int)
        elif self.rule_init == "grid":
            self.antecedents_ = grid_rules(X.shape[1], self.n_terms)
        elif self.rule_init == "fcm":
            self.antecedents_ = fcm_rules(Xn, target, self.centers_, self.sigmas_, self.n_rules, self.seed)
        else:
            raise ValueError("rule_init must be 'fcm' or 'grid'")

        self.history_ = {"train": [], "val": []}
        return self._train(X, target, X_val, target_val, self.epochs)

    def fine_tune(self, X, target, X_val=None, target_val=None, epochs=100):
        """Continue hybrid learning from the current parameters (e.g. after pruning).

        Terms still may not drift more than `max_shift` from their *original*
        positions, so fine-tuning cannot change what "Low" or "High" means.
        """
        return self._train(np.asarray(X, dtype=float), np.asarray(target, dtype=float),
                           X_val, target_val, epochs)

    def _train(self, X, target, X_val, target_val, epochs):
        Xn = self._scale(X)
        has_val = X_val is not None and target_val is not None
        if has_val:
            Xn_val = self._scale(X_val)
            target_val = np.asarray(target_val, dtype=float)

        optimizer = Adam(self.lr)
        best = (np.inf, None)
        for _ in range(epochs):
            wbar = self._normalize(self._log_firing(Xn))
            self.consequents_ = self._solve_consequents(wbar, target)        # forward pass
            y = wbar @ self.consequents_
            self.history_["train"].append(float(np.mean((y - target) ** 2)))
            if has_val:
                val_loss = float(np.mean((self._raw_output(Xn_val) - target_val) ** 2))
                self.history_["val"].append(val_loss)
                if val_loss < best[0]:
                    best = (val_loss, (self.centers_.copy(), self.sigmas_.copy(), self.consequents_.copy()))
            grad_c, grad_s = self._premise_gradients(Xn, wbar, y, target)   # backward pass
            optimizer.step([self.centers_, self.sigmas_], [grad_c, grad_s])
            self._apply_constraints()

        if best[1] is not None:
            self.centers_, self.sigmas_, self.consequents_ = best[1]
        else:
            self.refit_consequents(X, target)
        return self

    # ----------------------------------------------------------- inference
    def _raw_output(self, Xn):
        return self._normalize(self._log_firing(Xn)) @ self.consequents_

    def predict_proba(self, X):
        """Layer 5 output, read as P(diabetic)."""
        return np.clip(self._raw_output(self._scale(X)), 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def memberships(self, X):
        """Layer 1 output: (n, n_features, n_terms) membership degrees."""
        return np.exp(self._log_memberships(self._scale(X)))

    def firing_strengths(self, X):
        """Layers 2 and 3: raw firing strengths w and normalised weights wn."""
        log_w = self._log_firing(self._scale(X))
        return np.exp(log_w), self._normalize(log_w)

    # ------------------------------------------------- rule-base editing
    def refit_consequents(self, X, target):
        wbar = self._normalize(self._log_firing(self._scale(X)))
        self.consequents_ = self._solve_consequents(wbar, np.asarray(target, dtype=float))
        return self

    def with_antecedents(self, antecedents, X, target):
        """Copy of this model with a different rule base; consequents re-fitted."""
        model = copy.deepcopy(self)
        model.antecedents_ = np.asarray(antecedents, dtype=int)
        return model.refit_consequents(X, target)
