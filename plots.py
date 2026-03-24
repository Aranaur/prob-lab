# =============================================================================
# Pure matplotlib plotting functions — each takes data and returns a Figure
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import DARK_BG, dark_style


def draw_ci_plot(history_data: list[dict], mu: float, sigma: float, n: int,
                 method: str = "t"):
    """Horizontal confidence-interval chart (last ≤50 intervals)."""
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=DARK_BG)
    dark_style(ax)

    se_theory = sigma / np.sqrt(n)
    x_lo = mu - 5 * se_theory
    x_hi = mu + 5 * se_theory

    # True-mean reference line
    ax.axvline(mu, color="#f59e0b", linewidth=1.1, linestyle="--", zorder=5)

    # Method badge (top-right corner)
    method_label = {"t": "t-interval", "z": "z-interval", "bootstrap": "Bootstrap"}.get(method, method)
    ax.text(0.99, 0.99, method_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#94a3b8",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#1e293b",
                  "edgecolor": "#334155", "alpha": 0.8})

    if len(history_data) == 0:
        ax.text(mu, 0.5, "Press Sample or Play to begin",
                ha="center", va="center", color="#64748b", fontsize=12)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Sample mean", fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    for idx, entry in enumerate(history_data):
        y = idx + 1
        color = "#94a3b8" if entry["covered"] else "#f87171"
        ax.plot([entry["lower"], entry["upper"]], [y, y],
                color=color, linewidth=1.0, solid_capstyle="round", zorder=2)
        ax.plot(entry["mean"], y, "o",
                color="#38bdf8", markersize=4, alpha=0.9, zorder=3)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.5, len(history_data) + 0.5)
    ax.set_xlabel("Sample mean", fontsize=10)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def draw_prop_plot(px: list, py: list, conf_target: float):
    """Running proportion of CIs that include μ."""
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=DARK_BG)
    dark_style(ax)

    ax.axhline(conf_target, color="#38bdf8", linewidth=0.8,
               linestyle="--", zorder=3)

    if len(px) == 0:
        ax.set_xlim(0, 100)
        ax.set_ylim(max(0, conf_target - 0.2), 1.0)
    else:
        ax.plot(px, py, color="#e2e8f0", linewidth=0.9, zorder=4)
        y_min = max(0, min(min(py), conf_target) - 0.05)
        y_max = min(1, max(max(py), conf_target) + 0.05)
        ax.set_ylim(y_min, y_max)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Samples drawn", fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def draw_width_plot(widths: list):
    """Histogram + KDE of CI widths."""
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=DARK_BG)
    dark_style(ax)

    if len(widths) < 3:
        ax.text(0.5, 0.5, "Collecting data\u2026",
                ha="center", va="center", color="#64748b",
                fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    elif np.std(widths) < 1e-10:
        # z-interval: all widths are identical (constant CI width)
        w0 = widths[0]
        ax.axvline(w0, color="#818cf8", linewidth=2.0)
        ax.set_xlim(w0 * 0.9, w0 * 1.1)
        ax.set_xlabel("CI Width", fontsize=9)
        ax.set_yticks([])
        ax.text(0.56, 0.65,
                f"Constant width\n{w0:.4f}",
                ha="left", va="center", fontsize=9,
                color="#a5b4fc", transform=ax.transAxes)
    else:
        ax.hist(widths, bins="auto", density=True,
                color="#818cf8", alpha=0.5, edgecolor="#a5b4fc", linewidth=0.6)
        kde = gaussian_kde(widths)
        xs = np.linspace(min(widths), max(widths), 200)
        ax.plot(xs, kde(xs), color="#a5b4fc", linewidth=1.0)
        ax.set_xlabel("CI Width", fontsize=9)
        ax.set_yticks([])

    fig.tight_layout()
    plt.close(fig)
    return fig


def draw_means_plot(sample_means: list, mu: float, sigma: float, n: int):
    """Histogram of sample means with theoretical N(μ, σ/√n) overlay."""
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=DARK_BG)
    dark_style(ax)

    if len(sample_means) < 3:
        ax.text(0.5, 0.5, "Collecting data\u2026",
                ha="center", va="center", color="#64748b",
                fontsize=11, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(sample_means, bins="auto", density=True,
                color="#c084fc", alpha=0.5, edgecolor="#d8b4fe", linewidth=0.6)

        theo_se = sigma / np.sqrt(n)
        emp_mean = float(np.mean(sample_means))
        emp_se = float(np.std(sample_means, ddof=1))

        xs = np.linspace(min(sample_means), max(sample_means), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, theo_se),
                color="#d8b4fe", linewidth=1.2, linestyle="--",
                label=f"N(\u03bc, \u03c3/\u221an)")

        ax.axvline(mu, color="#f59e0b", linewidth=0.9, linestyle="--", zorder=5)
        ax.set_xlabel("Sample mean (\u0304x)", fontsize=9)
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right",
                  facecolor="#1e293b", edgecolor="#334155",
                  labelcolor="#cbd5e1")

        # Empirical vs theoretical stats (mathtext)
        line1 = rf"$\bar{{x}}$ = {emp_mean:+.3f}   ($\mu$ = {mu:+.3f})"
        line2 = rf"SE = {emp_se:.3f}   ($\sigma/\sqrt{{n}}$ = {theo_se:.3f})"
        ax.text(0.02, 0.97, line1 + "\n" + line2,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#cbd5e1",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#1e293b",
                      "edgecolor": "#475569", "alpha": 0.85})

    fig.tight_layout()
    plt.close(fig)
    return fig
