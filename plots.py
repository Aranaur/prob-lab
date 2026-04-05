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
# 1.  CI chart  (horizontal intervals)
# ═════════════════════════════════════════════════════════════════════════════
def draw_ci_plot(history_data: list[dict], true_val: float, sigma: float,
                 n: int, method: str = "t", statistic: str = "mean",
                 dark: bool = True) -> go.Figure:

    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    # x-axis range based on statistic
    if statistic == "mean":
        se_theory = sigma / np.sqrt(n)
        x_lo = true_val - 5 * se_theory
        x_hi = true_val + 5 * se_theory
        x_title = "Sample mean (x\u0304)"
        est_sym = "x\u0304"
        param_sym = "\u03bc"
    elif statistic == "median":
        se_theory = sigma / np.sqrt(n)
        x_lo = true_val - 5 * se_theory
        x_hi = true_val + 5 * se_theory
        x_title = "Sample median"
        est_sym = "Med"
        param_sym = "Median"
    else:  # variance
        spread = sigma ** 2 * 3 / np.sqrt(n)
        x_lo = max(0, true_val - spread)
        x_hi = true_val + spread
        x_title = "Sample variance (s\u00b2)"
        est_sym = "s\u00b2"
        param_sym = "\u03c3\u00b2"

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
    else:
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines",
            line=dict(color=t["line"], width=1.2),
            hovertemplate="n=%{x}<br>Coverage=%{y:.1%}<extra></extra>",
        ))
        y_min = max(0, min(min(py), conf_target) - 0.05)
        y_max = min(1, max(max(py), conf_target) + 0.05)
        fig.update_yaxes(range=[y_min, y_max])

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
                    n: int, statistic: str = "mean",
                    dark: bool = True) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    if statistic == "mean":
        x_title = "Sample mean (x\u0304)"
        est_sym, param_sym = "x\u0304", "\u03bc"
    elif statistic == "median":
        x_title = "Sample median"
        est_sym, param_sym = "Med", "Median"
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
