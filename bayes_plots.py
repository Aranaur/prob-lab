# =============================================================================
# bayes_plots.py — Plotly figures for the Bayesian Explorer
#
# Nine pure functions, three per sub-tab:
#   Tab 1 (Beta-Binomial):  prior_posterior, coin_sequence, evolution_forest
#   Tab 2 (Freq vs Bayes):  forest_comparison, width_hist, running_coverage
#   Tab 3 (A/B Testing):    ab_posteriors, ab_joint, ab_prob_evolution
# =============================================================================

from __future__ import annotations

import numpy as np
from scipy import stats
import plotly.graph_objects as go

from plots import _base_fig, _theme


# ── Palette (aligned with existing modules) ──────────────────────────────────
_C_PRIOR      = "#94a3b8"   # slate-400 — muted (past belief)
_C_LIKELIHOOD = "#f59e0b"   # amber-500 — data speaks
_C_POSTERIOR  = "#818cf8"   # indigo-400 — updated belief
_C_FREQ       = "#22d3ee"   # cyan-400
_C_BAYES      = "#c084fc"   # purple-400
_C_TRUE       = "#ef4444"   # red-500 — true parameter
_C_SUCCESS    = "#10b981"   # emerald-500 — H
_C_FAIL       = "#64748b"   # slate-500 — T
_C_A          = "#22d3ee"   # variant A
_C_B          = "#c084fc"   # variant B
_C_DIAG       = "#475569"   # y=x reference
_C_THRESH     = "#f59e0b"   # decision threshold


def _empty(fig: go.Figure, msg: str, dark: bool) -> go.Figure:
    t = _theme(dark)
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=msg, showarrow=False,
        font=dict(size=13, color=t["annot_text"]),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 — Beta-Binomial Updating
# ═════════════════════════════════════════════════════════════════════════════

def draw_bys1_prior_posterior(
    alpha_prior: float,
    beta_prior: float,
    k: int,
    n: int,
    dark: bool = True,
) -> go.Figure:
    """Prior · Likelihood · Posterior Beta densities on p ∈ [0, 1]."""
    fig = _base_fig(dark)
    t = _theme(dark)

    a_post = alpha_prior + k
    b_post = beta_prior + n - k

    # Likelihood (as a Beta(k+1, n-k+1) — proportional to the binomial likelihood in p)
    a_like = k + 1
    b_like = n - k + 1

    x = np.linspace(0.001, 0.999, 400)
    prior = stats.beta.pdf(x, alpha_prior, beta_prior)
    post  = stats.beta.pdf(x, a_post, b_post)
    like  = stats.beta.pdf(x, a_like, b_like) if n > 0 else np.zeros_like(x)

    fig.add_trace(go.Scatter(
        x=x, y=prior, mode="lines", name="Prior",
        line=dict(color=_C_PRIOR, width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(148,163,184,0.08)",
        hovertemplate="p=%{x:.3f}<br>prior=%{y:.2f}<extra></extra>",
    ))
    if n > 0:
        fig.add_trace(go.Scatter(
            x=x, y=like, mode="lines", name="Likelihood",
            line=dict(color=_C_LIKELIHOOD, width=2, dash="dash"),
            hovertemplate="p=%{x:.3f}<br>likelihood=%{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=post, mode="lines", name="Posterior",
        line=dict(color=_C_POSTERIOR, width=3),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.18)",
        hovertemplate="p=%{x:.3f}<br>posterior=%{y:.2f}<extra></extra>",
    ))

    # p̂ reference
    if n > 0:
        phat = k / n
        fig.add_vline(
            x=phat, line=dict(color=_C_LIKELIHOOD, width=1, dash="dot"),
            annotation_text=f"p\u0302 = {phat:.3f}", annotation_position="top",
            annotation_font=dict(size=10, color=_C_LIKELIHOOD),
        )

    # Legend row (manual annotations to keep style consistent with other modules)
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=(f"<span style='color:{_C_PRIOR}'>● Prior Beta({alpha_prior:g}, {beta_prior:g})</span>  "
              f"<span style='color:{_C_LIKELIHOOD}'>● Likelihood ({k}/{n})</span>  "
              f"<span style='color:{_C_POSTERIOR}'>● Posterior Beta({a_post:g}, {b_post:g})</span>"),
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1,
    )

    fig.update_layout(xaxis_title="p", yaxis_title="density",
                      xaxis=dict(range=[0, 1]))
    return fig


def draw_bys1_coin_sequence(obs: list[int], dark: bool = True) -> go.Figure:
    """Sequence of coin-flips with running p\u0302."""
    fig = _base_fig(dark)
    t = _theme(dark)

    if not obs:
        return _empty(fig, "Sample a few observations to see the sequence", dark)

    arr = np.asarray(obs, dtype=int)
    n = arr.size
    idx = np.arange(1, n + 1)
    running = np.cumsum(arr) / idx

    # Running p̂ line
    fig.add_trace(go.Scatter(
        x=idx, y=running, mode="lines",
        line=dict(color=_C_POSTERIOR, width=2),
        hovertemplate="n=%{x}<br>p\u0302=%{y:.3f}<extra></extra>",
    ))

    # H / T markers (on y=0 baseline, coloured by outcome)
    is_h = arr == 1
    fig.add_trace(go.Scatter(
        x=idx[is_h], y=np.zeros(is_h.sum()),
        mode="markers",
        marker=dict(color=_C_SUCCESS, size=6, symbol="circle",
                    line=dict(color=_C_SUCCESS, width=0)),
        hovertemplate="flip #%{x}: H<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=idx[~is_h], y=np.zeros((~is_h).sum()),
        mode="markers",
        marker=dict(color=_C_FAIL, size=6, symbol="circle-open",
                    line=dict(color=_C_FAIL, width=1.5)),
        hovertemplate="flip #%{x}: T<extra></extra>",
    ))

    final_phat = running[-1]
    fig.add_hline(
        y=final_phat,
        line=dict(color=_C_POSTERIOR, width=1, dash="dot"),
        annotation_text=f"p\u0302 = {final_phat:.3f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color=_C_POSTERIOR),
    )

    fig.update_layout(
        xaxis_title="flip #", yaxis_title="running p\u0302",
        yaxis=dict(range=[-0.05, 1.05]),
    )
    return fig


def draw_bys1_evolution_forest(
    alpha_prior: float,
    beta_prior: float,
    obs: list[int],
    dark: bool = True,
    level: float = 0.95,
) -> go.Figure:
    """Forest-plot: 95% CrI at checkpoints n = 1, 2, 5, 10, 20, 50, ..."""
    fig = _base_fig(dark)
    t = _theme(dark)

    if not obs:
        return _empty(fig, "Sample observations to watch the posterior shrink", dark)

    arr = np.asarray(obs, dtype=int)
    n_total = arr.size
    cum = np.cumsum(arr)

    # Checkpoints: 1, 2, 5, 10, 20, 50, 100, 200, 500, ..., + n_total
    ladder = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    checkpoints = sorted(set([c for c in ladder if c <= n_total] + [n_total]))

    a_lo = (1 - level) / 2
    a_hi = 1 - a_lo
    rows = []
    for n_ in checkpoints:
        k_ = int(cum[n_ - 1])
        a = alpha_prior + k_
        b = beta_prior + n_ - k_
        lo = float(stats.beta.ppf(a_lo, a, b))
        hi = float(stats.beta.ppf(a_hi, a, b))
        mean = a / (a + b)
        rows.append((n_, k_, lo, hi, mean))

    labels = [f"n = {r[0]}  (k={r[1]})" for r in rows]

    for i, (n_, k_, lo, hi, mean) in enumerate(rows):
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[labels[i], labels[i]],
            mode="lines",
            line=dict(color=_C_POSTERIOR, width=3),
            hovertemplate=f"n={n_}  k={k_}<br>CrI=[{lo:.3f}, {hi:.3f}]<br>width={hi - lo:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[mean], y=[labels[i]], mode="markers",
            marker=dict(color=_C_POSTERIOR, size=9, symbol="diamond"),
            hoverinfo="skip",
        ))
        fig.add_annotation(
            x=hi, y=labels[i], text=f"  w={hi - lo:.3f}",
            showarrow=False, xanchor="left",
            font=dict(size=9, color=t["muted"]),
        )

    fig.update_layout(
        xaxis_title="p", xaxis=dict(range=[0, 1]),
        yaxis=dict(type="category", autorange="reversed"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — Frequentist vs Bayesian CI
# ═════════════════════════════════════════════════════════════════════════════

def draw_bys2_forest(
    history: list[tuple],
    true_p: float,
    dark: bool = True,
    max_show: int = 50,
) -> go.Figure:
    """Side-by-side forest plot of last ``max_show`` experiments.

    Each `history` entry: (freq_lo, freq_hi, bayes_lo, bayes_hi, cov_f, cov_b).
    """
    fig = _base_fig(dark)
    t = _theme(dark)

    if not history:
        return _empty(fig, "Sample to compare frequentist CI vs Bayesian CrI", dark)

    last = history[-max_show:]
    n = len(last)

    # Stack vertically: one row per experiment, freq on the left half, bayes on the right.
    # We use y = integer rows; offset freq/bayes by ±0.18.
    for i, (flo, fhi, blo, bhi, covf, covb) in enumerate(last):
        y = i
        col_f = _C_FREQ if covf else _C_TRUE
        col_b = _C_BAYES if covb else _C_TRUE
        fig.add_trace(go.Scatter(
            x=[flo, fhi], y=[y + 0.18, y + 0.18],
            mode="lines", line=dict(color=col_f, width=2),
            hovertemplate=f"freq #{i + 1}<br>[{flo:.3f}, {fhi:.3f}]<br>covered={covf}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[blo, bhi], y=[y - 0.18, y - 0.18],
            mode="lines", line=dict(color=col_b, width=2),
            hovertemplate=f"bayes #{i + 1}<br>[{blo:.3f}, {bhi:.3f}]<br>covered={covb}<extra></extra>",
        ))

    fig.add_vline(
        x=true_p, line=dict(color=_C_TRUE, width=1.5, dash="dash"),
        annotation_text=f"true p = {true_p:.3f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color=_C_TRUE),
    )

    # Legend row
    fig.add_annotation(
        x=0.02, y=1.04, xref="paper", yref="paper",
        text=(f"<span style='color:{_C_FREQ}'>─ Frequentist CI</span>    "
              f"<span style='color:{_C_BAYES}'>─ Bayesian CrI</span>    "
              f"<span style='color:{_C_TRUE}'>─ miss</span>"),
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color=t["annot_text"]),
    )

    fig.update_layout(
        xaxis_title="p", xaxis=dict(range=[0, 1]),
        yaxis=dict(title="experiment", tickmode="array",
                   tickvals=list(range(0, n, max(1, n // 10))),
                   ticktext=[str(i + 1) for i in range(0, n, max(1, n // 10))],
                   range=[-0.8, n - 0.2]),
    )
    return fig


def draw_bys2_width_hist(history: list[tuple], dark: bool = True) -> go.Figure:
    """Dual histogram of interval widths — freq vs bayes."""
    fig = _base_fig(dark)
    t = _theme(dark)

    if not history:
        return _empty(fig, "Sample to see the width distributions", dark)

    freq_w  = np.array([h[1] - h[0] for h in history])
    bayes_w = np.array([h[3] - h[2] for h in history])

    wmax = float(max(freq_w.max(), bayes_w.max()))
    bins = np.linspace(0, max(wmax * 1.05, 1e-6), 30)

    fig.add_trace(go.Histogram(
        x=freq_w, xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
        marker=dict(color=_C_FREQ, line=dict(width=0)), opacity=0.55,
        name="freq width",
        hovertemplate="width %{x:.3f}<br>count %{y}<extra>freq</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=bayes_w, xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
        marker=dict(color=_C_BAYES, line=dict(width=0)), opacity=0.55,
        name="bayes width",
        hovertemplate="width %{x:.3f}<br>count %{y}<extra>bayes</extra>",
    ))

    fig.add_vline(x=float(freq_w.mean()),
                  line=dict(color=_C_FREQ, width=1.5, dash="dot"))
    fig.add_vline(x=float(bayes_w.mean()),
                  line=dict(color=_C_BAYES, width=1.5, dash="dot"))

    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=(f"<span style='color:{_C_FREQ}'>freq  mean w = {freq_w.mean():.3f}</span><br>"
              f"<span style='color:{_C_BAYES}'>bayes mean w = {bayes_w.mean():.3f}</span>"),
        showarrow=False, xanchor="right", yanchor="top", align="right",
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1,
    )

    fig.update_layout(
        xaxis_title="interval width", yaxis_title="count",
        barmode="overlay",
    )
    return fig


def draw_bys2_running_coverage(
    history: list[tuple], nominal: float, dark: bool = True,
) -> go.Figure:
    """Running coverage % vs experiments, two lines (freq, bayes)."""
    fig = _base_fig(dark)
    t = _theme(dark)

    if not history:
        return _empty(fig, "Sample to watch coverage converge", dark)

    covf = np.array([h[4] for h in history], dtype=float)
    covb = np.array([h[5] for h in history], dtype=float)
    idx  = np.arange(1, covf.size + 1)
    run_f = np.cumsum(covf) / idx * 100
    run_b = np.cumsum(covb) / idx * 100

    fig.add_trace(go.Scatter(
        x=idx, y=run_f, mode="lines",
        line=dict(color=_C_FREQ, width=2),
        hovertemplate="n=%{x}<br>freq cov=%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=idx, y=run_b, mode="lines",
        line=dict(color=_C_BAYES, width=2),
        hovertemplate="n=%{x}<br>bayes cov=%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=nominal, line=dict(color=_C_TRUE, width=1.5, dash="dash"),
        annotation_text=f"nominal {nominal:.0f}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color=_C_TRUE),
    )

    fig.add_annotation(
        x=0.02, y=0.02, xref="paper", yref="paper",
        text=(f"<span style='color:{_C_FREQ}'>● freq {run_f[-1]:.1f}%</span>   "
              f"<span style='color:{_C_BAYES}'>● bayes {run_b[-1]:.1f}%</span>"),
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1,
    )

    fig.update_layout(
        xaxis_title="experiment #", yaxis_title="running coverage %",
        yaxis=dict(range=[0, 105]),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — Bayesian A/B Testing
# ═════════════════════════════════════════════════════════════════════════════

def draw_bys3_posteriors(
    aA: float, bA: float, aB: float, bB: float, dark: bool = True,
) -> go.Figure:
    """Overlay posterior densities for variants A and B."""
    fig = _base_fig(dark)
    t = _theme(dark)

    x = np.linspace(0.001, 0.999, 400)
    pdfA = stats.beta.pdf(x, aA, bA)
    pdfB = stats.beta.pdf(x, aB, bB)

    fig.add_trace(go.Scatter(
        x=x, y=pdfA, mode="lines", name="A",
        line=dict(color=_C_A, width=2),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.15)",
        hovertemplate="p=%{x:.3f}<br>A density=%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pdfB, mode="lines", name="B",
        line=dict(color=_C_B, width=2),
        fill="tozeroy", fillcolor="rgba(192,132,252,0.15)",
        hovertemplate="p=%{x:.3f}<br>B density=%{y:.2f}<extra></extra>",
    ))

    meanA = aA / (aA + bA)
    meanB = aB / (aB + bB)
    fig.add_vline(x=meanA, line=dict(color=_C_A, width=1, dash="dot"))
    fig.add_vline(x=meanB, line=dict(color=_C_B, width=1, dash="dot"))

    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=(f"<span style='color:{_C_A}'>● A  Beta({aA:.1f}, {bA:.1f})  mean {meanA:.3f}</span><br>"
              f"<span style='color:{_C_B}'>● B  Beta({aB:.1f}, {bB:.1f})  mean {meanB:.3f}</span>"),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1,
    )

    fig.update_layout(xaxis_title="p", yaxis_title="density",
                      xaxis=dict(range=[0, 1]))
    return fig


def draw_bys3_joint(
    pA: np.ndarray, pB: np.ndarray, dark: bool = True, max_points: int = 4000,
) -> go.Figure:
    """Scatter of joint posterior draws; points above y=x are 'B wins'."""
    fig = _base_fig(dark)
    t = _theme(dark)

    if pA.size == 0 or pB.size == 0:
        return _empty(fig, "Draw posterior samples to see the joint distribution", dark)

    if pA.size > max_points:
        idx = np.random.default_rng(0).choice(pA.size, max_points, replace=False)
        pA_s = pA[idx]; pB_s = pB[idx]
    else:
        pA_s, pB_s = pA, pB

    b_wins = pB_s > pA_s
    # Points where B wins → purple; A wins → cyan.
    fig.add_trace(go.Scattergl(
        x=pA_s[b_wins], y=pB_s[b_wins], mode="markers",
        marker=dict(color=_C_B, size=3, opacity=0.35),
        hovertemplate="pA=%{x:.3f}<br>pB=%{y:.3f}  (B wins)<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=pA_s[~b_wins], y=pB_s[~b_wins], mode="markers",
        marker=dict(color=_C_A, size=3, opacity=0.35),
        hovertemplate="pA=%{x:.3f}<br>pB=%{y:.3f}  (A wins)<extra></extra>",
    ))

    # y = x diagonal
    lo = float(min(pA_s.min(), pB_s.min()))
    hi = float(max(pA_s.max(), pB_s.max()))
    pad = (hi - lo) * 0.05 + 1e-3
    lo -= pad; hi += pad
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=_C_DIAG, width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    prob_b = float(np.mean(pB > pA))
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"P(B > A) = <b>{prob_b:.3f}</b>",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=12, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1,
    )

    fig.update_layout(
        xaxis_title="p_A", yaxis_title="p_B",
        xaxis=dict(range=[lo, hi]),
        yaxis=dict(range=[lo, hi], scaleanchor="x", scaleratio=1),
    )
    return fig


def draw_bys3_prob_evolution(
    trajectory: list[tuple[int, float]],
    threshold: float,
    dark: bool = True,
) -> go.Figure:
    """P(B > A) as a function of cumulative n (per sampling batch)."""
    fig = _base_fig(dark)
    t = _theme(dark)

    if not trajectory:
        return _empty(fig, "Sample batches to watch P(B>A) evolve", dark)

    ns    = [pt[0] for pt in trajectory]
    probs = [pt[1] for pt in trajectory]

    fig.add_trace(go.Scatter(
        x=ns, y=probs, mode="lines+markers",
        line=dict(color=_C_POSTERIOR, width=2),
        marker=dict(color=_C_POSTERIOR, size=5),
        hovertemplate="n=%{x}<br>P(B>A)=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=threshold, line=dict(color=_C_THRESH, width=1.5, dash="dash"),
        annotation_text=f"threshold {threshold:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color=_C_THRESH),
    )
    fig.add_hline(y=0.5, line=dict(color=t["muted"], width=1, dash="dot"))

    fig.update_layout(
        xaxis_title="cumulative n (per variant)",
        yaxis_title="P(B > A)",
        yaxis=dict(range=[0, 1]),
    )
    return fig
