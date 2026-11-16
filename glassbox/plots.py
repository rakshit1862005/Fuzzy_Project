"""Figures for the report. All plots share one light theme and colour order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .membership import log_memberships  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical: blue, orange, aqua
TERM_RAMP = ["#86b6ef", "#2a78d6", "#104281"]       # ordinal: Low -> High (one hue)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "font.family": "sans-serif",
    "lines.linewidth": 2,
    "figure.dpi": 110,
})


def _save(fig, path):
    """Save and close when a path is given; otherwise return the figure (for the app)."""
    fig.tight_layout()
    if path is None:
        return fig
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return None


def feature_importance(importance, path, selected_names=()):
    table = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = [SERIES[0] if f in selected_names else AXIS for f in table["feature"]]
    ax.barh(table["feature"], table["importance"], color=colors, height=0.6)
    for y, v in enumerate(table["importance"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Mean change in network output when the feature is shuffled")
    ax.set_title("What the network relies on (blue = given to the fuzzy system)", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def membership_functions(model, path, x=None):
    """One panel per input; `x` (original units) marks a patient's values."""
    k = len(model.feature_names)
    cols = 2
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 2.6 * rows), squeeze=False)
    for j, name in enumerate(model.feature_names):
        ax = axes[j // cols][j % cols]
        xs = np.linspace(0, 1, 400)
        x_orig = model.lo_[j] + xs * model.span_[j]
        mu0 = np.exp(log_memberships(xs[:, None], model.initial_centers_[j:j + 1], model.initial_sigmas_[j:j + 1]))[:, 0]
        mu = np.exp(log_memberships(xs[:, None], model.centers_[j:j + 1], model.sigmas_[j:j + 1]))[:, 0]
        for t, term in enumerate(model.term_names):
            color = TERM_RAMP[t] if model.n_terms == 3 else SERIES[t % 3]
            ax.plot(x_orig, mu0[:, t], color=color, lw=1, ls="--", alpha=0.7)
            ax.plot(x_orig, mu[:, t], color=color, lw=2, label=term)
        if x is not None:
            ax.axvline(x[j], color=SERIES[1], lw=1.5)
            ax.text(x[j], 1.04, f" patient: {x[j]:.4g}", color=INK_2, fontsize=8, va="bottom")
        ax.set_title(name, loc="left", pad=14 if x is not None else 6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("membership")
        ax.legend(loc="center right", ncol=1)
    for j in range(k, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle("Linguistic terms after training (dashed = initial)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    return _save(fig, path)


def training_curve(history, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color=SERIES[0], label="training")
    if history.get("val"):
        ax.plot(epochs, history["val"], color=SERIES[1], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error vs network")
    ax.set_title("ANFIS hybrid learning", loc="left")
    ax.legend()
    return _save(fig, path)


def tradeoff(summary, path):
    """summary: one row per configuration with method, rules_mean, conditions_mean,
    fidelity_mean and fidelity_std (spread over random seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    methods = list(dict.fromkeys(summary["method"]))
    for ax, xcol, xlabel in ((axes[0], "rules_mean", "number of rules"),
                             (axes[1], "conditions_mean", "total conditions (IF-parts) in the rule base")):
        for i, method in enumerate(methods):
            part = summary[summary["method"] == method]
            ax.errorbar(part[xcol], part["fidelity_mean"], yerr=part["fidelity_std"], color=SERIES[i],
                        fmt="o", ms=6, capsize=2, elinewidth=1, label=method,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
    axes[0].set_ylabel("fidelity to the network (test set)")
    fig.suptitle("Fidelity vs interpretability - each dot is one configuration; upper left is better",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    axes[1].legend(loc="lower right")
    return _save(fig, path)


def case_contributions(expl, path, top_k=6):
    rules = expl["rules"][:top_k][::-1]
    labels = [f"R{r['rule']}: " + (" & ".join(c["condition"] for c in r["conditions"]) or "(always)")
              for r in rules]
    weights = [r["weight"] for r in rules]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rules) + 1.4))
    bars = ax.barh(range(len(rules)), weights, color=SERIES[0], height=0.6)
    for bar, r in zip(bars, rules):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  weight {r['weight']:.2f}  x  risk {r['consequent']:.2f}",
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rules)), labels, fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("normalised firing strength (share of the decision)")
    nn = f"network {expl['nn_prob']:.2f}, " if "nn_prob" in expl else ""
    weak = "  (weak match)" if expl.get("weak_match") else ""
    fig.suptitle(f"Why: {nn}fuzzy risk {expl['fuzzy_output']:.2f} -> {expl['fuzzy_decision']}{weak}",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
