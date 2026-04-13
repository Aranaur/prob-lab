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

    # Use t-distribution for t-tests, normal for z-test
    if test_type == "one_z" or df is None:
        y0 = stats.norm.pdf(xs, mu0, se)
        y1 = stats.norm.pdf(xs, mu1, se)
    else:
        y0 = stats.t.pdf((xs - mu0) / se, df) / se
        y1 = stats.t.pdf((xs - mu1) / se, df) / se

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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Population-level overlap & Cohen's d visualisation
# ─────────────────────────────────────────────────────────────────────────────
def draw_cohens_d_overlap(
    d: float,
    dark: bool = True,
) -> go.Figure:
    """Two population distributions N(0,1) vs N(d,1) with overlap shading."""
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    d_abs = abs(d)
    mu0, mu1 = 0.0, d_abs          # always show H₁ to the right

    # Overlap area = 2·Φ(−|d|/2) for equal-variance normals
    overlap_pct = float(2 * stats.norm.cdf(-d_abs / 2) * 100)

    # Effect-size interpretation (Cohen, 1988)
    if d_abs < 0.2:
        eff_label, eff_color = "negligible", "#94a3b8"
    elif d_abs < 0.5:
        eff_label, eff_color = "small", "#34d399"
    elif d_abs < 0.8:
        eff_label, eff_color = "medium", "#fbbf24"
    else:
        eff_label, eff_color = "large", "#f87171"

    # X range — symmetric around midpoint
    spread = max(d_abs + 3.5, 3.5)
    centre = (mu0 + mu1) / 2
    xs = np.linspace(centre - spread, centre + spread, 500)

    y0 = stats.norm.pdf(xs, mu0, 1.0)
    y1 = stats.norm.pdf(xs, mu1, 1.0)
    y_overlap = np.minimum(y0, y1)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(
                text="Standardised units (σ)",
                font=dict(size=10, color=t["label"]),
            ),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # ── Overlap fill ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_overlap, np.zeros(len(xs))]),
        fill="toself", fillcolor="rgba(103,232,249,0.25)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
    ))

    # ── Curves ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=y0, mode="lines",
        line=dict(color="#94a3b8", width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=y1, mode="lines",
        line=dict(color="#818cf8", width=1.5), hoverinfo="skip",
    ))

    # ── Mean reference lines ─────────────────────────────────────────────────
    fig.add_vline(x=mu0, line_dash="dash", line_color="#94a3b8", line_width=1)
    if d_abs > 1e-9:
        fig.add_vline(x=mu1, line_dash="dash", line_color="#818cf8", line_width=1)

    # ── Curve labels ─────────────────────────────────────────────────────────
    y0_peak = float(y0.max())
    fig.add_annotation(
        x=mu0, y=y0_peak * 1.08, text="H\u2080",
        showarrow=False, font=dict(size=11, color="#94a3b8"),
    )
    if d_abs > 1e-9:
        fig.add_annotation(
            x=mu1, y=float(y1.max()) * 1.08, text="H\u2081",
            showarrow=False, font=dict(size=11, color="#818cf8"),
        )

    # ── Double-headed arrow showing d ────────────────────────────────────────
    if d_abs > 0.05:
        arrow_y = y0_peak * 0.82
        # →
        fig.add_annotation(
            x=mu1, y=arrow_y, ax=mu0, ay=arrow_y,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
            arrowcolor="#38bdf8",
        )
        # ←
        fig.add_annotation(
            x=mu0, y=arrow_y, ax=mu1, ay=arrow_y,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
            arrowcolor="#38bdf8",
        )
        # label
        fig.add_annotation(
            x=centre, y=arrow_y * 1.12,
            text=f"d\u200a=\u200a{d_abs:.3f}",
            showarrow=False, font=dict(size=10, color="#38bdf8"),
        )

    # ── Stats annotation ─────────────────────────────────────────────────────
    lines = [
        f"Cohen\u2019s d\u200a=\u200a{d_abs:.3f}",
        f"Overlap\u200a=\u200a{overlap_pct:.1f}\u200a%",
        f"Effect: <span style='color:{eff_color}'>{eff_label}</span>",
    ]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text="<br>".join(lines), showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4, align="left",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  H₀ / H₁ sampling distributions for proportions
# ─────────────────────────────────────────────────────────────────────────────
def draw_prop_distributions(
    p0: float,
    p1: float,
    n: int,
    alpha: float,
    power: float,
    alternative: str,
    dark: bool = True,
) -> go.Figure:
    """Sampling distributions of p̂ under H₀ (p=p₀) and H₁ (p=p₁)."""
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    # SE under H₀ (pooled) and H₁ (unpooled)
    p_pool = (p0 + p1) / 2
    se_h0 = np.sqrt(p_pool * (1 - p_pool) * 2 / max(n, 2))
    se_h1 = np.sqrt(p0 * (1 - p0) / max(n, 2) + p1 * (1 - p1) / max(n, 2))

    # Guard against degenerate SE
    if se_h0 < 1e-12 or se_h1 < 1e-12:
        fig = _base_fig(dark=dark)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="SE \u2248 0 \u2014 increase n or change p\u2080/p\u2081",
            showarrow=False, font=dict(size=13, color=t["muted"]),
        )
        return fig

    spread = max(abs(p1 - p0) + 4.5 * max(se_h0, se_h1), 5.0 * se_h0)
    centre = (p0 + p1) / 2
    xs = np.linspace(centre - spread, centre + spread, 500)

    y0 = stats.norm.pdf(xs, p0, se_h0)
    y1 = stats.norm.pdf(xs, p1, se_h1)

    # Critical values on p̂ scale
    if alternative == "two-sided":
        z = stats.norm.ppf(1 - alpha / 2)
        cv_hi = p0 + z * se_h0
        cv_lo = p0 - z * se_h0
        cv_lines = [cv_lo, cv_hi]
        alpha_masks = [xs <= cv_lo, xs >= cv_hi]
        power_masks = [xs <= cv_lo, xs >= cv_hi]
        beta_mask = (xs >= cv_lo) & (xs <= cv_hi)
    elif alternative == "greater":
        z = stats.norm.ppf(1 - alpha)
        cv_hi = p0 + z * se_h0
        cv_lines = [cv_hi]
        alpha_masks = [xs >= cv_hi]
        power_masks = [xs >= cv_hi]
        beta_mask = xs < cv_hi
    else:
        z = stats.norm.ppf(alpha)
        cv_lo = p0 + z * se_h0
        cv_lines = [cv_lo]
        alpha_masks = [xs <= cv_lo]
        power_masks = [xs <= cv_lo]
        beta_mask = xs > cv_lo

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Sample proportion (p\u0302)", font=dict(size=10, color=t["label"])),
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

    # ── Reference lines for p₀ and p₁ ────────────────────────────────────────
    fig.add_vline(x=p0, line_dash="dash", line_color="#94a3b8", line_width=1)
    if abs(p1 - p0) > 1e-9:
        fig.add_vline(x=p1, line_dash="dash", line_color="#818cf8", line_width=1)

    # ── Curve labels ──────────────────────────────────────────────────────────
    fig.add_annotation(
        x=p0, y=float(max(y0)) * 1.07, text="H\u2080",
        showarrow=False, font=dict(size=11, color="#94a3b8"),
    )
    if abs(p1 - p0) > 1e-9:
        fig.add_annotation(
            x=p1, y=float(max(y1)) * 1.07, text="H\u2081",
            showarrow=False, font=dict(size=11, color="#818cf8"),
        )

    # ── Stats annotation ─────────────────────────────────────────────────────
    delta_p = p1 - p0
    beta = max(0.0, 1 - power)
    muted_col = t["muted"]
    lines = [
        f"p\u2080\u200a=\u200a{p0:.3f}",
        f"p\u2081\u200a=\u200a{p1:.3f}",
        f"\u0394p\u200a=\u200a{delta_p:+.4f}",
        f"Power\u200a=\u200a{power:.3f}",
        f"\u03b2\u200a=\u200a{beta:.3f}",
        f"\u03b1\u200a=\u200a{alpha}",
        f"n\u200a=\u200a{n:,}",
        f"<span style='color:{muted_col}'>two-prop Z</span>",
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
# 5.  Proportion effect visualisation (analogue of Cohen's d overlap)
# ─────────────────────────────────────────────────────────────────────────────
def draw_prop_effect(
    p0: float,
    p1: float,
    dark: bool = True,
) -> go.Figure:
    """Two population Bernoulli distributions shown as normal approximations
    N(p, sqrt(p(1-p))) with overlap shading and Cohen's h annotation."""
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    sigma0 = np.sqrt(max(p0 * (1 - p0), 1e-12))
    sigma1 = np.sqrt(max(p1 * (1 - p1), 1e-12))

    delta_p = p1 - p0
    rel_lift = delta_p / p0 * 100 if p0 > 1e-9 else 0.0
    h = float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0)))
    h_abs = abs(h)

    # Effect-size interpretation (Cohen, 1988 — same thresholds as d)
    if h_abs < 0.2:
        eff_label, eff_color = "negligible", "#94a3b8"
    elif h_abs < 0.5:
        eff_label, eff_color = "small", "#34d399"
    elif h_abs < 0.8:
        eff_label, eff_color = "medium", "#fbbf24"
    else:
        eff_label, eff_color = "large", "#f87171"

    # X range
    spread = max(abs(delta_p) + 3.5 * max(sigma0, sigma1), 3.5 * max(sigma0, sigma1))
    centre = (p0 + p1) / 2
    xs = np.linspace(centre - spread, centre + spread, 500)

    y0 = stats.norm.pdf(xs, p0, sigma0)
    y1 = stats.norm.pdf(xs, p1, sigma1)
    y_overlap = np.minimum(y0, y1)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Proportion", font=dict(size=10, color=t["label"])),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # ── Overlap fill ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_overlap, np.zeros(len(xs))]),
        fill="toself", fillcolor="rgba(103,232,249,0.25)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
    ))

    # ── Curves ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=y0, mode="lines",
        line=dict(color="#94a3b8", width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=y1, mode="lines",
        line=dict(color="#818cf8", width=1.5), hoverinfo="skip",
    ))

    # ── Mean reference lines ─────────────────────────────────────────────────
    fig.add_vline(x=p0, line_dash="dash", line_color="#94a3b8", line_width=1)
    if abs(delta_p) > 1e-9:
        fig.add_vline(x=p1, line_dash="dash", line_color="#818cf8", line_width=1)

    # ── Curve labels ─────────────────────────────────────────────────────────
    y0_peak = float(y0.max())
    fig.add_annotation(
        x=p0, y=y0_peak * 1.08, text="H\u2080",
        showarrow=False, font=dict(size=11, color="#94a3b8"),
    )
    if abs(delta_p) > 1e-9:
        fig.add_annotation(
            x=p1, y=float(y1.max()) * 1.08, text="H\u2081",
            showarrow=False, font=dict(size=11, color="#818cf8"),
        )

    # ── Double-headed arrow showing Δp ───────────────────────────────────────
    if abs(delta_p) > 0.001:
        arrow_y = y0_peak * 0.82
        fig.add_annotation(
            x=p1, y=arrow_y, ax=p0, ay=arrow_y,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1.5, arrowcolor="#38bdf8",
        )
        fig.add_annotation(
            x=p0, y=arrow_y, ax=p1, ay=arrow_y,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1.5, arrowcolor="#38bdf8",
        )
        fig.add_annotation(
            x=centre, y=arrow_y * 1.12,
            text=f"\u0394p\u200a=\u200a{delta_p:+.4f}",
            showarrow=False, font=dict(size=10, color="#38bdf8"),
        )

    # ── Stats annotation ─────────────────────────────────────────────────────
    lines = [
        f"Cohen\u2019s h\u200a=\u200a{h_abs:.3f}",
        f"\u0394p\u200a=\u200a{delta_p:+.4f}",
        f"Rel. lift\u200a=\u200a{rel_lift:+.1f}\u200a%",
        f"Effect: <span style='color:{eff_color}'>{eff_label}</span>",
    ]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text="<br>".join(lines), showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4, align="left",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  H₀ / H₁ sampling distributions for ratio metrics (Delta Method)
# ─────────────────────────────────────────────────────────────────────────────
def draw_ratio_distributions(
    r0: float,
    r1: float,
    n: int,
    alpha: float,
    power: float,
    alternative: str,
    var_r: float,
    dark: bool = True,
) -> go.Figure:
    """Sampling distributions of R̂ under H₀ (R=R₀) and H₁ (R=R₁)
    using Delta Method variance.  SE = √(2·Var(R)/n) for two-sample."""
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    # SE of the difference (two-sample, equal groups)
    se = np.sqrt(2 * var_r / max(n, 2))

    if se < 1e-12:
        fig = _base_fig(dark=dark)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="SE \u2248 0 \u2014 check variance parameters",
            showarrow=False, font=dict(size=13, color=t["muted"]),
        )
        return fig

    spread = max(abs(r1 - r0) + 4.5 * se, 5.0 * se)
    centre = (r0 + r1) / 2
    xs = np.linspace(centre - spread, centre + spread, 500)

    y0 = stats.norm.pdf(xs, r0, se)
    y1 = stats.norm.pdf(xs, r1, se)

    # Critical values on R̂ scale
    if alternative == "two-sided":
        z = stats.norm.ppf(1 - alpha / 2)
        cv_hi = r0 + z * se
        cv_lo = r0 - z * se
        cv_lines = [cv_lo, cv_hi]
        alpha_masks = [xs <= cv_lo, xs >= cv_hi]
        power_masks = [xs <= cv_lo, xs >= cv_hi]
        beta_mask = (xs >= cv_lo) & (xs <= cv_hi)
    elif alternative == "greater":
        z = stats.norm.ppf(1 - alpha)
        cv_hi = r0 + z * se
        cv_lines = [cv_hi]
        alpha_masks = [xs >= cv_hi]
        power_masks = [xs >= cv_hi]
        beta_mask = xs < cv_hi
    else:
        z = stats.norm.ppf(alpha)
        cv_lo = r0 + z * se
        cv_lines = [cv_lo]
        alpha_masks = [xs <= cv_lo]
        power_masks = [xs <= cv_lo]
        beta_mask = xs > cv_lo

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Sample ratio (R\u0302)", font=dict(size=10, color=t["label"])),
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

    # ── Reference lines for R₀ and R₁ ────────────────────────────────────────
    fig.add_vline(x=r0, line_dash="dash", line_color="#94a3b8", line_width=1)
    if abs(r1 - r0) > 1e-9:
        fig.add_vline(x=r1, line_dash="dash", line_color="#818cf8", line_width=1)

    # ── Curve labels ──────────────────────────────────────────────────────────
    fig.add_annotation(
        x=r0, y=float(max(y0)) * 1.07, text="H\u2080",
        showarrow=False, font=dict(size=11, color="#94a3b8"),
    )
    if abs(r1 - r0) > 1e-9:
        fig.add_annotation(
            x=r1, y=float(max(y1)) * 1.07, text="H\u2081",
            showarrow=False, font=dict(size=11, color="#818cf8"),
        )

    # ── Stats annotation ─────────────────────────────────────────────────────
    delta_r = r1 - r0
    lift_pct = delta_r / r0 * 100 if abs(r0) > 1e-9 else 0.0
    beta_val = max(0.0, 1 - power)
    muted_col = t["muted"]
    lines = [
        f"R\u2080\u200a=\u200a{r0:.4f}",
        f"R\u2081\u200a=\u200a{r1:.4f}",
        f"\u0394R\u200a=\u200a{delta_r:+.5f}",
        f"Lift\u200a=\u200a{lift_pct:+.2f}\u200a%",
        f"Power\u200a=\u200a{power:.3f}",
        f"\u03b2\u200a=\u200a{beta_val:.3f}",
        f"\u03b1\u200a=\u200a{alpha}",
        f"n\u200a=\u200a{n:,}",
        f"<span style='color:{muted_col}'>Delta method Z</span>",
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
# 7.  Ratio effect visualisation (Delta Method analogue of Cohen's d overlap)
# ─────────────────────────────────────────────────────────────────────────────
def draw_ratio_effect(
    r0: float,
    r1: float,
    var_r: float,
    dark: bool = True,
) -> go.Figure:
    """Two population-level ratio distributions N(R, √Var(R)) with overlap."""
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    sigma = np.sqrt(max(var_r, 1e-12))
    delta_r = r1 - r0
    lift_pct = delta_r / r0 * 100 if abs(r0) > 1e-9 else 0.0

    # Standardised effect for the ratio
    d_ratio = abs(delta_r) / sigma if sigma > 1e-9 else 0.0

    if d_ratio < 0.2:
        eff_label, eff_color = "negligible", "#94a3b8"
    elif d_ratio < 0.5:
        eff_label, eff_color = "small", "#34d399"
    elif d_ratio < 0.8:
        eff_label, eff_color = "medium", "#fbbf24"
    else:
        eff_label, eff_color = "large", "#f87171"

    # X range
    spread = max(abs(delta_r) + 3.5 * sigma, 3.5 * sigma)
    centre = (r0 + r1) / 2
    xs = np.linspace(centre - spread, centre + spread, 500)

    y0 = stats.norm.pdf(xs, r0, sigma)
    y1 = stats.norm.pdf(xs, r1, sigma)
    y_overlap = np.minimum(y0, y1)

    # Overlap percentage (equal-variance normals)
    overlap_pct = float(2 * stats.norm.cdf(-d_ratio / 2) * 100)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Ratio (X/Y)", font=dict(size=10, color=t["label"])),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # ── Overlap fill ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([y_overlap, np.zeros(len(xs))]),
        fill="toself", fillcolor="rgba(103,232,249,0.25)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
    ))

    # ── Curves ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=xs, y=y0, mode="lines",
        line=dict(color="#94a3b8", width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=y1, mode="lines",
        line=dict(color="#818cf8", width=1.5), hoverinfo="skip",
    ))

    # ── Mean reference lines ─────────────────────────────────────────────────
    fig.add_vline(x=r0, line_dash="dash", line_color="#94a3b8", line_width=1)
    if abs(delta_r) > 1e-9:
        fig.add_vline(x=r1, line_dash="dash", line_color="#818cf8", line_width=1)

    # ── Curve labels ─────────────────────────────────────────────────────────
    y0_peak = float(y0.max())
    fig.add_annotation(
        x=r0, y=y0_peak * 1.08, text="H\u2080",
        showarrow=False, font=dict(size=11, color="#94a3b8"),
    )
    if abs(delta_r) > 1e-9:
        fig.add_annotation(
            x=r1, y=float(y1.max()) * 1.08, text="H\u2081",
            showarrow=False, font=dict(size=11, color="#818cf8"),
        )

    # ── Double-headed arrow showing ΔR ───────────────────────────────────────
    if abs(delta_r) > sigma * 0.01:
        arrow_y = y0_peak * 0.82
        fig.add_annotation(
            x=r1, y=arrow_y, ax=r0, ay=arrow_y,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1.5, arrowcolor="#38bdf8",
        )
        fig.add_annotation(
            x=r0, y=arrow_y, ax=r1, ay=arrow_y,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1.5, arrowcolor="#38bdf8",
        )
        fig.add_annotation(
            x=centre, y=arrow_y * 1.12,
            text=f"\u0394R\u200a=\u200a{delta_r:+.5f}",
            showarrow=False, font=dict(size=10, color="#38bdf8"),
        )

    # ── Stats annotation ─────────────────────────────────────────────────────
    lines = [
        f"\u0394R/\u03c3\u200a=\u200a{d_ratio:.3f}",
        f"\u0394R\u200a=\u200a{delta_r:+.5f}",
        f"Rel. lift\u200a=\u200a{lift_pct:+.2f}\u200a%",
        f"Overlap\u200a=\u200a{overlap_pct:.1f}\u200a%",
        f"Effect: <span style='color:{eff_color}'>{eff_label}</span>",
    ]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text="<br>".join(lines), showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4, align="left",
    )

    return fig
