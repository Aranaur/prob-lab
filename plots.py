# =============================================================================
# Plotly chart builders — each returns a plotly.graph_objects.Figure
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

# ── Shared theme constants ────────────────────────────────────────────────────
_TRANSPARENT = "rgba(0,0,0,0)"

# Dark mode tokens
_GRID  = "#334155"
_AXIS  = "#94a3b8"
_LABEL = "#cbd5e1"

# Light mode tokens
_LIGHT_GRID  = "#e2e8f0"
_LIGHT_AXIS  = "#64748b"
_LIGHT_LABEL = "#475569"

_DARK_LAYOUT = dict(
    paper_bgcolor=_TRANSPARENT,
    plot_bgcolor=_TRANSPARENT,
    font=dict(family="Inter, sans-serif", color=_LABEL, size=11),
    margin=dict(l=40, r=12, t=8, b=36),
    xaxis=dict(
        gridcolor=_GRID, gridwidth=0.3, linecolor=_GRID,
        tickfont=dict(size=10, color=_AXIS), zeroline=False,
    ),
    yaxis=dict(
        gridcolor=_GRID, gridwidth=0.3, linecolor=_GRID,
        tickfont=dict(size=10, color=_AXIS), zeroline=False,
    ),
    showlegend=False,
    dragmode=False,
)

_LIGHT_LAYOUT = dict(
    paper_bgcolor=_TRANSPARENT,
    plot_bgcolor=_TRANSPARENT,
    font=dict(family="Inter, sans-serif", color=_LIGHT_LABEL, size=11),
    margin=dict(l=40, r=12, t=8, b=36),
    xaxis=dict(
        gridcolor=_LIGHT_GRID, gridwidth=0.3, linecolor=_LIGHT_GRID,
        tickfont=dict(size=10, color=_LIGHT_AXIS), zeroline=False,
    ),
    yaxis=dict(
        gridcolor=_LIGHT_GRID, gridwidth=0.3, linecolor=_LIGHT_GRID,
        tickfont=dict(size=10, color=_LIGHT_AXIS), zeroline=False,
    ),
    showlegend=False,
    dragmode=False,
)

_CONFIG = dict(displayModeBar=False, staticPlot=False)


def _theme(dark: bool) -> dict:
    """Per-draw annotation/text colour tokens."""
    if dark:
        return dict(
            label=_LABEL, axis=_AXIS, grid=_GRID,
            annot_bg="#1e293b", annot_border="#334155", annot_border2="#475569",
            annot_text="#cbd5e1", muted="#64748b",
            line="#e2e8f0",
        )
    return dict(
        label=_LIGHT_LABEL, axis=_LIGHT_AXIS, grid=_LIGHT_GRID,
        annot_bg="#ffffff", annot_border="#c7d2fe", annot_border2="#a5b4fc",
        annot_text="#1e293b", muted="#94a3b8",
        line="#334155",
    )


def _base_fig(dark: bool = True, **overrides) -> go.Figure:
    """Create a Figure with the shared theme layout."""
    base = _DARK_LAYOUT if dark else _LIGHT_LAYOUT
    layout = {**base, **overrides}
    return go.Figure(layout=layout)


# ═════════════════════════════════════════════════════════════════════════════
# 0.  Population distribution with sample rug plot
# ═════════════════════════════════════════════════════════════════════════════
def draw_population_plot(
    dist: str,
    params: dict,
    sample: list,
    true_val: float,
    statistic: str = "mean",
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    is_discrete = dist in ("poisson", "binomial")

    # ── x range and PDF/PMF ──────────────────────────────────────────────────
    if dist == "normal":
        mu_, sg_ = params.get("mu", 0.0), params.get("sigma", 1.0)
        x_lo, x_hi = mu_ - 4.5 * sg_, mu_ + 4.5 * sg_
        xs  = np.linspace(x_lo, x_hi, 400)
        ys  = stats.norm.pdf(xs, mu_, sg_)

    elif dist == "uniform":
        a_, b_ = params.get("a", 0.0), params.get("b", 1.0)
        pad = (b_ - a_) * 0.3
        x_lo, x_hi = a_ - pad, b_ + pad
        xs  = np.linspace(x_lo, x_hi, 400)
        ys  = stats.uniform.pdf(xs, a_, b_ - a_)

    elif dist == "exponential":
        lam_ = params.get("lam", 1.0)
        scale_ = 1.0 / max(lam_, 1e-9)
        x_lo, x_hi = 0.0, stats.expon.ppf(0.995, scale=scale_)
        xs  = np.linspace(x_lo, x_hi, 400)
        ys  = stats.expon.pdf(xs, scale=scale_)

    elif dist == "lognormal":
        lnmu_, lnsg_ = params.get("lnmu", 0.0), params.get("lnsg", 0.5)
        x_lo = 0.0
        x_hi = stats.lognorm.ppf(0.99, s=lnsg_, scale=np.exp(lnmu_))
        xs   = np.linspace(1e-9, x_hi, 400)
        ys   = stats.lognorm.pdf(xs, s=lnsg_, scale=np.exp(lnmu_))

    elif dist == "poisson":
        lam_ = params.get("lam", 3.0)
        k_lo = max(0, int(lam_ - 4 * np.sqrt(lam_) - 1))
        k_hi = int(lam_ + 4 * np.sqrt(lam_) + 2)
        xs   = np.arange(k_lo, k_hi + 1, dtype=float)
        ys   = stats.poisson.pmf(xs.astype(int), lam_)
        x_lo, x_hi = k_lo - 0.5, k_hi + 0.5

    elif dist == "binomial":
        m_, p_ = params.get("m", 10), params.get("p", 0.5)
        xs   = np.arange(0, m_ + 1, dtype=float)
        ys   = stats.binom.pmf(xs.astype(int), m_, p_)
        x_lo, x_hi = -0.5, m_ + 0.5

    else:
        xs, ys = np.array([0.0, 1.0]), np.array([1.0, 1.0])
        x_lo, x_hi = 0.0, 1.0

    rug_y   = -max(ys) * 0.08          # rug ticks sit just below the x-axis
    y_lo    = rug_y * 1.6
    y_hi    = max(ys) * 1.18

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, range=[x_lo, x_hi],
                   title=dict(text="x", font=dict(size=10, color=t["label"]))),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[y_lo, y_hi]),
        margin=dict(l=40, r=12, t=6, b=32),
    )

    # ── Distribution curve / bars ────────────────────────────────────────────
    if is_discrete:
        for xi, yi in zip(xs, ys):
            fig.add_shape(type="line",
                x0=xi, x1=xi, y0=0, y1=yi,
                line=dict(color="rgba(129,140,248,0.55)", width=6))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color="#818cf8", size=6),
            hovertemplate="k=%{x:.0f}<br>P=%{y:.4f}<extra></extra>",
        ))
    else:
        # Filled area under the curve
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([ys, np.zeros(len(xs))]),
            fill="toself", fillcolor="rgba(129,140,248,0.12)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="#818cf8", width=1.8),
            hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.4f}<extra></extra>",
        ))

    # ── Sample rug ticks ─────────────────────────────────────────────────────
    if len(sample) > 0:
        samp = np.array(sample)
        fig.add_trace(go.Scatter(
            x=samp,
            y=np.full(len(samp), rug_y),
            mode="markers",
            marker=dict(
                symbol="line-ns", size=10,
                color="#38bdf8", opacity=0.7,
                line=dict(color="#38bdf8", width=1.2),
            ),
            hovertemplate="x=%{x:.3f}<extra></extra>",
            name="sample",
        ))

        # ── Sample statistic — dot on the x-axis ─────────────────────────────
        if statistic != "variance":
            if statistic == "mean":
                s_val = float(np.mean(samp))
            elif statistic == "median":
                s_val = float(np.median(samp))
            elif statistic == "percentile":
                s_val = float(np.percentile(samp, params.get("p_level", 25)))
            else:
                s_val = None

            if s_val is not None:
                fig.add_trace(go.Scatter(
                    x=[s_val], y=[rug_y],
                    mode="markers",
                    marker=dict(color="#38bdf8", size=7,
                                line=dict(color=t["annot_text"], width=1.5)),
                    hovertemplate=f"{s_val:.3f}<extra></extra>",
                ))

    # ── True parameter line (no label — axis alignment is sufficient) ────────
    if statistic != "variance":
        fig.add_vline(x=true_val, line_dash="dash",
                      line_color="#f59e0b", line_width=1.2)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 1.  CI chart  (horizontal intervals)
# ═════════════════════════════════════════════════════════════════════════════
def draw_ci_plot(history_data: list[dict], true_val: float, sigma: float,
                 n: int, method: str = "t", statistic: str = "mean",
                 p_level: int = 25, dark: bool = True) -> go.Figure:

    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    se_theory = sigma / np.sqrt(n)

    # x-axis range and labels based on statistic
    if statistic == "mean":
        x_lo, x_hi = true_val - 5 * se_theory, true_val + 5 * se_theory
        x_title, est_sym, param_sym = "Sample mean (x\u0304)", "x\u0304", "\u03bc"
    elif statistic == "median":
        x_lo, x_hi = true_val - 5 * se_theory, true_val + 5 * se_theory
        x_title, est_sym, param_sym = "Sample median", "Med", "Median"
    elif statistic == "percentile":
        x_lo, x_hi = true_val - 5 * se_theory, true_val + 5 * se_theory
        x_title = f"Sample P{p_level}"
        est_sym  = f"P{p_level}"
        param_sym = f"P{p_level} (pop.)"
    else:  # variance
        spread = sigma ** 2 * 3 / np.sqrt(n)
        x_lo, x_hi = max(0, true_val - spread), true_val + spread
        x_title, est_sym, param_sym = "Sample variance (s\u00b2)", "s\u00b2", "\u03c3\u00b2"

    method_label = {"t": "t-interval", "z": "z-interval",
                    "bootstrap": "Bootstrap"}.get(method, method)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, range=[x_lo, x_hi],
                   title=dict(text=x_title, font=dict(size=11, color=t["label"]))),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   linecolor=t["grid"]),
    )

    # True-parameter reference line
    fig.add_vline(x=true_val, line_dash="dash", line_color="#f59e0b", line_width=1.2)

    if len(history_data) == 0:
        fig.add_annotation(
            x=true_val, y=0.5, xref="x", yref="paper",
            text="Press Sample or Play to begin",
            showarrow=False, font=dict(size=13, color=t["muted"]),
        )
        fig.update_yaxes(range=[0, 1])
        return fig

    # Auto-fit x range to data
    all_lo = min(e["lower"] for e in history_data)
    all_hi = max(e["upper"] for e in history_data)
    margin = (all_hi - all_lo) * 0.15 if all_hi > all_lo else abs(true_val) * 0.5
    x_lo = min(x_lo, all_lo - margin)
    x_hi = max(x_hi, all_hi + margin)
    fig.update_xaxes(range=[x_lo, x_hi])

    for idx, entry in enumerate(history_data):
        y = idx + 1
        color = "#94a3b8" if entry["covered"] else "#f87171"
        fig.add_shape(
            type="line", x0=entry["lower"], x1=entry["upper"], y0=y, y1=y,
            line=dict(color=color, width=1.4), layer="above",
        )
        fig.add_trace(go.Scatter(
            x=[entry["estimate"]], y=[y], mode="markers",
            marker=dict(color="#38bdf8", size=5),
            hovertemplate=(
                f"{est_sym} = {entry['estimate']:.3f}<br>"
                f"CI = [{entry['lower']:.3f},  {entry['upper']:.3f}]<br>"
                f"Width = {entry['width']:.3f}<br>"
                f"{'Covers' if entry['covered'] else 'Misses'} {param_sym}"
                "<extra></extra>"
            ),
        ))

    fig.update_yaxes(range=[0.3, len(history_data) + 0.7])

    # Method badge
    fig.add_annotation(
        xref="paper", yref="paper", x=1, y=1, xanchor="right", yanchor="top",
        text=method_label, showarrow=False,
        font=dict(size=10, color=t["axis"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
    )

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Proportion of CIs including μ
# ═════════════════════════════════════════════════════════════════════════════
def draw_prop_plot(px: list, py: list, conf_target: float,
                   dark: bool = True) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="Samples drawn", font=dict(size=10, color=t["label"]))),
        yaxis=dict(**_ay, tickformat=".0%"),
    )

    fig.add_hline(y=conf_target, line_dash="dash", line_color="#38bdf8", line_width=0.9)

    if len(px) == 0:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[max(0, conf_target - 0.2), 1.0])
        return fig

    ns   = np.array(px, dtype=float)
    phat = np.array(py, dtype=float)

    # Wilson score CI for the empirical coverage proportion
    z = stats.norm.ppf(1 - (1 - conf_target) / 2)
    z2 = z ** 2
    denom   = ns + z2
    p_tilde = (phat + z2 / (2 * ns)) / (1 + z2 / ns)
    margin  = z * np.sqrt(ns * phat * (1 - phat) + z2 / 4) / denom
    lo = np.clip(p_tilde - margin, 0, 1)
    hi = np.clip(p_tilde + margin, 0, 1)

    # Shaded Wilson CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([ns, ns[::-1]]),
        y=np.concatenate([hi, lo[::-1]]),
        fill="toself",
        fillcolor="rgba(148,163,184,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=f"Wilson {conf_target:.0%} CI",
    ))

    # Empirical coverage line
    fig.add_trace(go.Scatter(
        x=px, y=py, mode="lines",
        line=dict(color=t["line"], width=1.2),
        hovertemplate=(
            "n=%{x}<br>"
            "Coverage=%{y:.1%}<extra></extra>"
        ),
    ))

    y_min = max(0.0, float(min(lo.min(), conf_target)) - 0.03)
    y_max = min(1.0, float(max(hi.max(), conf_target)) + 0.03)
    fig.update_yaxes(range=[y_min, y_max])

    # Annotation — top-right corner, Wilson CI for current coverage
    last_p, last_lo, last_hi = phat[-1], lo[-1], hi[-1]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        text=f"Coverage: {last_p:.1%}<br>Wilson {conf_target:.0%} CI: [{last_lo:.1%}, {last_hi:.1%}]",
        showarrow=False,
        font=dict(size=9, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=4,
        align="left",
    )

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 3.  CI width distribution
# ═════════════════════════════════════════════════════════════════════════════
def draw_width_plot(widths: list, dark: bool = True) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="CI Width", font=dict(size=10, color=t["label"]))),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if len(widths) < 3:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    if np.std(widths) < 1e-10:
        w0 = widths[0]
        fig.add_vline(x=w0, line_color="#818cf8", line_width=2.5)
        fig.update_xaxes(range=[w0 * 0.9, w0 * 1.1])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.62, y=0.7,
            text=f"Constant width<br>{w0:.4f}",
            showarrow=False, font=dict(size=11, color="#a5b4fc"),
            align="left",
        )
        return fig

    fig.add_trace(go.Histogram(
        x=widths, histnorm="probability density",
        marker=dict(color="rgba(129,140,248,0.5)", line=dict(color="#a5b4fc", width=0.6)),
        hovertemplate="Width=%{x:.3f}<br>Density=%{y:.3f}<extra></extra>",
    ))

    kde = gaussian_kde(widths)
    xs = np.linspace(min(widths), max(widths), 200)
    fig.add_trace(go.Scatter(
        x=xs, y=kde(xs), mode="lines",
        line=dict(color="#a5b4fc", width=1.2),
        hoverinfo="skip",
    ))

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Sample means distribution (CLT)
# ═════════════════════════════════════════════════════════════════════════════
def draw_means_plot(sample_stats: list, true_val: float, sigma: float,
                    n: int, statistic: str = "mean", p_level: int = 25,
                    dark: bool = True) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    if statistic == "mean":
        x_title = "Sample mean (x\u0304)"
        est_sym, param_sym = "x\u0304", "\u03bc"
    elif statistic == "median":
        x_title = "Sample median"
        est_sym, param_sym = "Med", "Median"
    elif statistic == "percentile":
        x_title = f"Sample P{p_level}"
        est_sym, param_sym = f"P{p_level}", f"P{p_level} (pop.)"
    else:
        x_title = "Sample variance (s\u00b2)"
        est_sym, param_sym = "s\u00b2", "\u03c3\u00b2"

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text=x_title, font=dict(size=10, color=t["label"]))),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if len(sample_stats) < 3:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    fig.add_trace(go.Histogram(
        x=sample_stats, histnorm="probability density",
        marker=dict(color="rgba(192,132,252,0.5)",
                    line=dict(color="#d8b4fe", width=0.6)),
        hovertemplate=f"{est_sym}=%{{x:.3f}}<br>Density=%{{y:.3f}}<extra></extra>",
    ))

    # Theoretical CLT overlay — only for mean
    if statistic == "mean":
        theo_se = sigma / np.sqrt(n)
        xs = np.linspace(min(sample_stats), max(sample_stats), 200)
        fig.add_trace(go.Scatter(
            x=xs, y=stats.norm.pdf(xs, true_val, theo_se),
            mode="lines", line=dict(color="#d8b4fe", width=1.4, dash="dash"),
            hoverinfo="skip",
        ))

    fig.add_vline(x=true_val, line_dash="dash", line_color="#f59e0b", line_width=1)

    emp_mean = float(np.mean(sample_stats))
    emp_se = float(np.std(sample_stats, ddof=1))

    if statistic == "mean":
        theo_se = sigma / np.sqrt(n)
        annot_text = (
            f"{est_sym} = {emp_mean:+.3f}  ({param_sym} = {true_val:+.3f})<br>"
            f"SE = {emp_se:.3f}  (\u03c3/\u221an = {theo_se:.3f})"
        )
    else:
        annot_text = (
            f"{est_sym} = {emp_mean:.3f}  ({param_sym} = {true_val:.3f})<br>"
            f"SE = {emp_se:.3f}"
        )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text=annot_text,
        showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"], borderwidth=1, borderpad=4,
        align="left",
    )

    return fig
