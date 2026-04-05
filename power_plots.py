# =============================================================================
# Power Explorer — Plotly chart builders
# =============================================================================

import numpy as np
from scipy import stats
import plotly.graph_objects as go

from plots import _DARK_LAYOUT, _LIGHT_LAYOUT, _base_fig, _theme


# ─────────────────────────────────────────────────────────────────────────────
# 1.  H₀ / H₁ sampling distributions with α, β, power regions
# ─────────────────────────────────────────────────────────────────────────────
def draw_power_distributions(
    d: float,
    n: int,
    se: float,
    alpha: float,
    power: float,
    alternative: str,
    test_type: str,
    df: int | None,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    # Means on the standardised x̄ scale (σ = 1)
    mu0 = 0.0
    mu1 = -d if alternative == "less" else d

    spread = max(abs(mu1) + 4.5 * se, 5.0 * se)
    center = (mu0 + mu1) / 2
    xs = np.linspace(center - spread, center + spread, 500)

    y0 = stats.norm.pdf(xs, mu0, se)
    y1 = stats.norm.pdf(xs, mu1, se)

    # Critical value(s) on x̄ scale
    def _cv(p):
        if test_type == "one_z":
            return float(stats.norm.ppf(p))
        return float(stats.t.ppf(p, max(df, 1)))

    if alternative == "two-sided":
        cv_hi = _cv(1 - alpha / 2) * se
        cv_lo = -cv_hi
        cv_lines = [cv_lo, cv_hi]
        alpha_masks = [xs <= cv_lo, xs >= cv_hi]
        power_masks = [xs <= cv_lo, xs >= cv_hi]
        beta_mask = (xs >= cv_lo) & (xs <= cv_hi)
    elif alternative == "greater":
        cv_hi = _cv(1 - alpha) * se
        cv_lines = [cv_hi]
        alpha_masks = [xs >= cv_hi]
        power_masks = [xs >= cv_hi]
        beta_mask = xs < cv_hi
    else:
        cv_lo = _cv(alpha) * se
        cv_lines = [cv_lo]
        alpha_masks = [xs <= cv_lo]
        power_masks = [xs <= cv_lo]
        beta_mask = xs > cv_lo

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Sample mean (x\u0304)", font=dict(size=10, color=t["label"])),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # ── Curves ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=y0, mode="lines",
        line=dict(color="#94a3b8", width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=y1, mode="lines",
        line=dict(color="#818cf8", width=1.5), hoverinfo="skip",
    ))

    # ── α region (under H₀) ──────────────────────────────────────────────────
    for mask in alpha_masks:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([xs[mask], xs[mask][::-1]]),
                y=np.concatenate([y0[mask], np.zeros(mask.sum())]),
                fill="toself", fillcolor="rgba(248,113,113,0.30)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            ))

    # ── Power region (under H₁) ──────────────────────────────────────────────
    for mask in power_masks:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([xs[mask], xs[mask][::-1]]),
                y=np.concatenate([y1[mask], np.zeros(mask.sum())]),
                fill="toself", fillcolor="rgba(129,140,248,0.40)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            ))

    # ── β region (under H₁, acceptance zone) ─────────────────────────────────
    if beta_mask.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs[beta_mask], xs[beta_mask][::-1]]),
            y=np.concatenate([y1[beta_mask], np.zeros(beta_mask.sum())]),
            fill="toself", fillcolor="rgba(148,163,184,0.15)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        ))

    # ── Critical value lines ──────────────────────────────────────────────────
    for cv in cv_lines:
        fig.add_vline(x=cv, line_dash="dot", line_color="#f87171", line_width=0.9)

    # ── Reference lines for μ₀ and μ₁ ────────────────────────────────────────
    fig.add_vline(x=mu0, line_dash="dash", line_color="#94a3b8", line_width=1)
    if abs(d) > 1e-9:
        fig.add_vline(x=mu1, line_dash="dash", line_color="#818cf8", line_width=1)

    # ── Curve labels ──────────────────────────────────────────────────────────
    fig.add_annotation(
        x=mu0, y=float(max(y0)) * 1.07, text="H\u2080",
        showarrow=False, font=dict(size=11, color="#94a3b8"),
    )
    if abs(d) > 1e-9:
        fig.add_annotation(
            x=mu1, y=float(max(y1)) * 1.07, text="H\u2081",
            showarrow=False, font=dict(size=11, color="#818cf8"),
        )

    # ── Stats annotation ─────────────────────────────────────────────────────
    beta = max(0.0, 1 - power)
    test_label = {"one_z": "Z", "one_t": "t", "two_t": "Welch t", "paired_t": "Paired t"}.get(test_type, "")
    muted_col = t["muted"]
    lines = [
        f"d\u200a=\u200a{d:.3f}",
        f"Power\u200a=\u200a{power:.3f}",
        f"\u03b2\u200a=\u200a{beta:.3f}",
        f"\u03b1\u200a=\u200a{alpha}",
        f"n\u200a=\u200a{n}",
        f"<span style='color:{muted_col}'>{test_label}-test</span>",
    ]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text="<br>".join(lines), showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4, align="left",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Power curve — power as a function of sample size
# ─────────────────────────────────────────────────────────────────────────────
def draw_power_curve(
    ns: np.ndarray,
    powers: np.ndarray,
    current_n: int,
    current_power: float,
    alpha: float,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Sample size (n)", font=dict(size=10, color=t["label"])),
        ),
        yaxis=dict(
            **_ay,
            title=dict(text="Power (1\u2212\u03b2)", font=dict(size=10, color=t["label"])),
            range=[0, 1.05], tickformat=".0%",
        ),
    )

    if len(ns) == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Set d\u200a>\u200a0 to see the power curve",
            showarrow=False, font=dict(size=13, color=t["muted"]),
        )
        return fig

    # Power curve
    fig.add_trace(go.Scatter(
        x=ns.tolist(), y=powers.tolist(), mode="lines",
        line=dict(color="#818cf8", width=2),
        hovertemplate="n=%{x}<br>Power=%{y:.3f}<extra></extra>",
    ))

    # α reference line
    fig.add_hline(y=alpha, line_dash="dot", line_color="#f87171", line_width=0.8)
    fig.add_annotation(
        x=0.02, y=alpha, xref="paper", yref="y",
        text=f"\u03b1={alpha}", showarrow=False,
        font=dict(size=9, color="#f87171"),
        xanchor="left", yanchor="bottom", yshift=2,
    )

    # 80 % convention line
    fig.add_hline(y=0.8, line_dash="dash", line_color="#38bdf8", line_width=0.7, opacity=0.5)

    # Current operating point
    fig.add_trace(go.Scatter(
        x=[current_n], y=[current_power], mode="markers",
        marker=dict(color="#38bdf8", size=9,
                    line=dict(color=t["annot_text"], width=1.5)),
        hovertemplate=f"n\u200a=\u200a{current_n}<br>Power\u200a=\u200a{current_power:.3f}<extra></extra>",
    ))

    # Crosshair from operating point
    fig.add_shape(
        type="line", x0=current_n, x1=current_n, y0=0, y1=current_power,
        line=dict(color="#38bdf8", width=1, dash="dot"),
    )
    fig.add_shape(
        type="line", x0=float(ns[0]), x1=current_n, y0=current_power, y1=current_power,
        line=dict(color="#38bdf8", width=1, dash="dot"),
    )

    return fig
