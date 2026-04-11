# =============================================================================
# Variance Reduction Explorer — Plotly chart helpers
# =============================================================================

import numpy as np
import plotly.graph_objects as go

# ── colour tokens ──────────────────────────────────────────────────────────────
_C_NAIVE  = "#f87171"   # red   — naive estimator
_C_METHOD = "#38bdf8"   # cyan  — CUPED / VWE
_C_CTRL   = "#94a3b8"   # slate — control group
_C_TREAT  = "#34d399"   # green — treatment group
_C_REG    = "#38bdf8"   # cyan  — regular users
_C_POWER  = "#f97316"   # orange — power users
_C_GOLD   = "#fbbf24"   # gold  — annotations

_DARK_BG    = "#0f172a"
_DARK_PAPER = "rgba(30,41,59,0.60)"
_DARK_GRID  = "rgba(148,163,184,0.08)"
_DARK_TEXT  = "#cbd5e1"

_LIGHT_BG    = "#ffffff"
_LIGHT_PAPER = "rgba(241,245,249,0.85)"
_LIGHT_GRID  = "rgba(100,116,139,0.10)"
_LIGHT_TEXT  = "#334155"


def _base_fig(dark: bool = True) -> go.Figure:
    bg    = _DARK_BG    if dark else _LIGHT_BG
    paper = _DARK_PAPER if dark else _LIGHT_PAPER
    grid  = _DARK_GRID  if dark else _LIGHT_GRID
    txt   = _DARK_TEXT  if dark else _LIGHT_TEXT
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=paper, plot_bgcolor=bg,
        font=dict(color=txt, size=11),
        margin=dict(l=48, r=16, t=10, b=40),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CUPED charts
# ═══════════════════════════════════════════════════════════════════════════════

def draw_cuped_scatter(x_ctrl, y_ctrl, x_treat, y_treat,
                       theta, delta_hat, dark=True):
    """Scatter X vs Y with two parallel regression lines (slope = θ̂)."""
    fig = _base_fig(dark)
    ann_col = _DARK_TEXT if dark else _LIGHT_TEXT

    # Subsample for rendering performance
    _MAX = 300
    for xs, ys, col, name in [
        (x_ctrl, y_ctrl, _C_CTRL, "Control"),
        (x_treat, y_treat, _C_TREAT, "Treatment"),
    ]:
        n = len(xs)
        idx = np.random.choice(n, min(_MAX, n), replace=False) if n > _MAX else np.arange(n)
        fig.add_trace(go.Scatter(
            x=xs[idx], y=ys[idx], mode="markers",
            marker=dict(color=col, size=4, opacity=0.35), name=name,
        ))

    # Parallel regression lines (same slope θ̂, different intercepts)
    all_x = np.concatenate([x_ctrl, x_treat])
    xr = np.array([np.percentile(all_x, 2), np.percentile(all_x, 98)])
    x_bar = all_x.mean()
    y_bar_c, y_bar_t = y_ctrl.mean(), y_treat.mean()

    for yb, col in [(y_bar_c, _C_CTRL), (y_bar_t, _C_TREAT)]:
        fig.add_trace(go.Scatter(
            x=xr, y=yb + theta * (xr - x_bar), mode="lines",
            line=dict(color=col, width=2.5), showlegend=False,
        ))

    # Vertical arrow showing δ̂_cuped
    fig.add_annotation(
        ax=x_bar, ay=y_bar_c, x=x_bar, y=y_bar_t,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowcolor=_C_GOLD, arrowwidth=2,
    )
    fig.add_annotation(
        x=x_bar, y=(y_bar_c + y_bar_t) / 2,
        text=f"δ̂<sub>cuped</sub>={delta_hat:.3f}",
        showarrow=False, font=dict(size=11, color=ann_col),
        xanchor="left", xshift=12,
    )

    fig.update_layout(
        xaxis_title="X (pre-experiment covariate)",
        yaxis_title="Y (outcome)",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig


def draw_cuped_variance_bars(var_naive, var_cuped, rho, n, dark=True):
    """Bar chart: Var(Δ̂) naive vs CUPED with reduction + equivalent-N annotation."""
    fig = _base_fig(dark)
    ann_col = _DARK_TEXT if dark else _LIGHT_TEXT
    vals = [var_naive, var_cuped]

    fig.add_trace(go.Bar(
        x=["Naive", "CUPED"], y=vals,
        marker_color=[_C_NAIVE, _C_METHOD], width=0.5,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))

    reduction = (1 - rho ** 2) * 100
    n_equiv = n / (1 - rho ** 2) if rho < 1 else float("inf")
    fig.add_annotation(
        x=0.5, y=max(vals) * 1.22, xref="paper",
        text=f"−{reduction:.0f}%  (≡ {n_equiv:.0f} obs/group)",
        showarrow=False, font=dict(size=11, color=ann_col),
    )

    fig.update_layout(
        yaxis_title="Var(Δ̂)", showlegend=False,
        yaxis=dict(rangemode="tozero"),
    )
    return fig


def draw_cuped_power_curve(rho_range, power_naive, power_cuped,
                           current_rho, dark=True):
    """Theoretical power vs ρ for naive and CUPED estimators."""
    fig = _base_fig(dark)
    fig.add_trace(go.Scatter(
        x=rho_range, y=power_naive, mode="lines",
        line=dict(color=_C_NAIVE, width=2, dash="dash"), name="Naive",
    ))
    fig.add_trace(go.Scatter(
        x=rho_range, y=power_cuped, mode="lines",
        line=dict(color=_C_METHOD, width=2.5), name="CUPED",
    ))
    fig.add_vline(
        x=current_rho,
        line=dict(color=_C_GOLD, width=1.5, dash="dot"),
        annotation_text=f"ρ={current_rho:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="ρ (covariate correlation)",
        yaxis_title="Power",
        yaxis=dict(range=[0, 1.05]),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# VWE charts
# ═══════════════════════════════════════════════════════════════════════════════

def draw_vwe_population(y_reg, y_pow, dark=True):
    """Mixed distribution histogram: regular vs power users."""
    fig = _base_fig(dark)
    if len(y_reg) > 0:
        fig.add_trace(go.Histogram(
            x=y_reg, marker_color=_C_REG, opacity=0.5,
            name="Regular", nbinsx=50,
        ))
    if len(y_pow) > 0:
        fig.add_trace(go.Histogram(
            x=y_pow, marker_color=_C_POWER, opacity=0.5,
            name="Power users", nbinsx=50,
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Y (outcome)", yaxis_title="Count",
        showlegend=True,
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    return fig


def draw_vwe_variance_decomp(pct_regular, sigma_reg, sigma_pow, dark=True):
    """Stacked bar: variance contribution in naive vs influence share in VWE."""
    fig = _base_fig(dark)
    p = pct_regular / 100
    var_total = p * sigma_reg ** 2 + (1 - p) * sigma_pow ** 2
    share_reg = p * sigma_reg ** 2 / var_total
    share_pow = 1 - share_reg

    # Naive variance share
    fig.add_trace(go.Bar(
        x=["Naive"], y=[share_reg], marker_color=_C_REG,
        name=f"Regular ({pct_regular:.0f}%)",
        text=[f"{share_reg * 100:.1f}%"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=["Naive"], y=[share_pow], marker_color=_C_POWER,
        name=f"Power ({100 - pct_regular:.0f}%)",
        text=[f"{share_pow * 100:.1f}%"], textposition="inside",
    ))

    # VWE effective influence (based on weights ∝ 1/σ²)
    w_r = 1 / sigma_reg ** 2
    w_p = 1 / sigma_pow ** 2
    w_tot = w_r + w_p
    vwe_r, vwe_p = w_r / w_tot, w_p / w_tot

    fig.add_trace(go.Bar(
        x=["VWE"], y=[vwe_r], marker_color=_C_REG, showlegend=False,
        text=[f"{vwe_r * 100:.1f}%"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=["VWE"], y=[vwe_p], marker_color=_C_POWER, showlegend=False,
        text=[f"{vwe_p * 100:.1f}%"], textposition="inside",
    ))

    fig.update_layout(
        barmode="stack",
        yaxis_title="Influence share",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        showlegend=True,
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    )
    return fig


def draw_vwe_ci_bars(ci_naive, ci_vwe, dark=True):
    """Bar chart: average CI width — naive vs VWE."""
    fig = _base_fig(dark)
    vals = [ci_naive, ci_vwe]
    fig.add_trace(go.Bar(
        x=["Naive", "VWE"], y=vals,
        marker_color=[_C_NAIVE, _C_METHOD], width=0.5,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    if ci_naive > 0:
        ann_col = _DARK_TEXT if dark else _LIGHT_TEXT
        fig.add_annotation(
            x=0.5, y=max(vals) * 1.18, xref="paper",
            text=f"Reduction: {(1 - ci_vwe / ci_naive) * 100:.1f}%",
            showarrow=False, font=dict(size=11, color=ann_col),
        )
    fig.update_layout(
        yaxis_title="Mean CI width", showlegend=False,
        yaxis=dict(rangemode="tozero"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Shared p-value histogram
# ═══════════════════════════════════════════════════════════════════════════════

def draw_vr_pvalue_hist(pv_naive, pv_method, alpha,
                        method_label="CUPED", dark=True):
    """Dual overlaid p-value histogram: naive vs CUPED / VWE."""
    fig = _base_fig(dark)
    bins_cfg = dict(start=0, end=1, size=0.025)

    if len(pv_naive) >= 5:
        fig.add_trace(go.Histogram(
            x=pv_naive, xbins=bins_cfg,
            marker_color=_C_NAIVE, opacity=0.5, name="Naive",
        ))
    if len(pv_method) >= 5:
        fig.add_trace(go.Histogram(
            x=pv_method, xbins=bins_cfg,
            marker_color=_C_METHOD, opacity=0.5, name=method_label,
        ))

    fig.add_vline(
        x=alpha,
        line=dict(color=_C_GOLD, width=1.5, dash="dash"),
        annotation_text=f"α={alpha}",
        annotation_position="top right",
    )

    ya = 0.98
    for pv, label, col in [(pv_naive, "Naive", _C_NAIVE),
                            (pv_method, method_label, _C_METHOD)]:
        if len(pv) > 0:
            rej = np.mean(np.array(pv) < alpha) * 100
            fig.add_annotation(
                x=0.98, y=ya, xref="paper", yref="paper",
                text=f"{label} reject: {rej:.1f}%",
                showarrow=False, font=dict(size=11, color=col),
                xanchor="right", yanchor="top",
            )
            ya -= 0.08

    fig.update_layout(
        barmode="overlay",
        xaxis_title="p-value", yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig
