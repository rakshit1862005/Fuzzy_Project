"""Figures for the project guide (docs/guide/figures/).

    python scripts/make_guide_figures.py

"Real" figures use the trained main model (report settings, seed 42) and the
actual data; "illustrative" figures are teaching sketches and say so in their
titles.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glassbox import config, plots  # noqa: E402  (importing plots sets the figure style)
from glassbox.membership import log_memberships  # noqa: E402
from glassbox.pipeline import train_glassbox  # noqa: E402

plt = plots.plt
OUT = config.ROOT / "docs" / "guide" / "figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def crisp_vs_fuzzy(model):
    """Real parameters: the trained 'Glucose is High' term vs a crisp cut-off at 140."""
    j = model.feature_names.index("Glucose")
    xs = np.linspace(model.lo_[j], model.hi_[j], 400)
    xn = (xs - model.lo_[j]) / model.span_[j]
    mu = np.exp(log_memberships(xn[:, None], model.centers_[j:j + 1], model.sigmas_[j:j + 1]))[:, 0, 2]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.step(xs, (xs >= 140).astype(float), where="post", color=plots.SERIES[1], label="crisp set: glucose ≥ 140")
    ax.plot(xs, mu, color=plots.SERIES[0], label="fuzzy set 'High' (trained model)")
    for g in (130, 150):
        m = float(np.interp(g, xs, mu))
        ax.plot([g], [m], "o", color=plots.SERIES[0])
        ax.annotate(f"{g} mg/dL → {m:.2f}", (g, m), textcoords="offset points", xytext=(6, -14), fontsize=8,
                    color=plots.INK_2)
    ax.set_xlabel("plasma glucose (mg/dL)")
    ax.set_ylabel("membership in 'High'")
    ax.set_title("Crisp vs fuzzy 'high glucose' (real model parameters)", loc="left")
    ax.legend(loc="center left")
    save(fig, "crisp_vs_fuzzy.png")


def mf_shapes():
    """Illustrative: common membership-function shapes on a 0-10 scale."""
    x = np.linspace(0, 10, 500)
    tri = np.clip(np.minimum((x - 2) / 3, (8 - x) / 3), 0, 1)
    trap = np.clip(np.minimum((x - 1) / 2, (9 - x) / 2), 0, 1)
    gauss = np.exp(-0.5 * ((x - 5) / 1.5) ** 2)
    shoulder = np.exp(-0.5 * (np.minimum(x - 7, 0) / 1.5) ** 2)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    for ax, y, title in zip(axes, (tri, trap, gauss, shoulder),
                            ("triangular", "trapezoidal", "Gaussian (used: Medium)", "Gaussian shoulder (used: High)")):
        ax.plot(x, y, color=plots.SERIES[0])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("membership")
    fig.suptitle("Membership-function shapes (illustrative)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, "mf_shapes.png")


def temperature():
    """Illustrative: 28 °C is partly Warm and partly Hot."""
    x = np.linspace(5, 45, 400)
    cold = np.clip((25 - x) / 10, 0, 1)
    warm = np.clip(np.minimum((x - 15) / 10, (35 - x) / 10), 0, 1)
    hot = np.clip((x - 25) / 10, 0, 1)
    fig, ax = plt.subplots(figsize=(7, 3))
    for y, name, col in ((cold, "Cold", plots.TERM_RAMP[0]), (warm, "Warm", plots.TERM_RAMP[1]),
                         (hot, "Hot", plots.TERM_RAMP[2])):
        ax.plot(x, y, color=col, label=name)
    ax.axvline(28, color=plots.SERIES[1], lw=1.5)
    ax.plot([28, 28], [0.7, 0.3], "o", color=plots.SERIES[1])
    ax.annotate("28 °C: Warm 0.7, Hot 0.3", (28, 0.7), textcoords="offset points", xytext=(8, 4), fontsize=8.5,
                color=plots.INK_2)
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("membership")
    ax.set_title("A value can belong to two fuzzy sets at once (illustrative)", loc="left")
    ax.legend(loc="center right")
    save(fig, "temperature.png")


def nn_loss(blackbox):
    """Real: the network's training loss per epoch."""
    mlp = blackbox.pipeline.named_steps["mlp"]
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color=plots.SERIES[0])
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy + L2 penalty")
    ax.set_title(f"Neural network training loss (real run, stopped at epoch {mlp.n_iter_})", loc="left")
    save(fig, "nn_loss.png")


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((config.RESULTS_DIR / "nn_params.json").read_text(encoding="utf-8"))
    gb = train_glassbox(nn_params=params)
    model = gb.surrogate.model
    crisp_vs_fuzzy(model)
    mf_shapes()
    temperature()
    nn_loss(gb.blackbox)
    x152 = gb.data.X_test[152, gb.surrogate.selected]
    plots.membership_functions(model, OUT / "patient152_memberships.png", x=x152)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
