# =============================================================================
# GoF Explorer — Plotly chart builders
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

from plots import _DARK_LAYOUT, _LIGHT_LAYOUT, _base_fig, _theme


# ── Colour tokens ────────────────────────────────────────────────────────────
_ECDF1   = "#818cf8"
_ECDF2   = "#34d399"
_CDF     = "#94a3b8"
_OBS     = "#818cf8"
_EXP     = "#94a3b8"
_REJECT  = "#f87171"
_STAT    = "#38bdf8"
_QQ_PT   = "#818cf8"
_QQ_REF  = "#f87171"
_QQ_BAND = "rgba(248,113,113,0.12)"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ecdf(sample):
    xs = np.sort(sample)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  KS one-sample — Fₙ(x) vs F₀(x)
# ─────────────────────────────────────────────────────────────────────────────
def draw_ks1_ecdf(
    sample: np.ndarray,
    dist_frozen,
    d_stat: float,
    d_loc: float,
    alpha: float,
    alternative: str,
    dist_label: str,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    xs, ecdf_y = _ecdf(sample)
    n = len(xs)
    pad = max((xs[-1] - xs[0]) * 0.08, 0.5)
    x_lo, x_hi = float(xs[0] - pad), float(xs[-1] + pad)

    x_sm = np.linspace(x_lo, x_hi, 400)
    cdf_sm = dist_frozen.cdf(x_sm)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="x", font=dict(size=10, color=t["label"]))),
        yaxis=dict(**_ay, title=dict(text="F(x)", font=dict(size=10, color=t["label"])),
                   range=[-0.03, 1.07]),
    )

    # Theoretical CDF
    fig.add_trace(go.Scatter(
        x=x_sm, y=cdf_sm, mode="lines",
        line=dict(color=_CDF, width=1.5, dash="dot"), hoverinfo="skip",
    ))

    # Empirical CDF (step)
    step_x = np.concatenate([[x_lo], xs, [x_hi]])
    step_y = np.concatenate([[0.0], ecdf_y, [1.0]])
    fig.add_trace(go.Scatter(
        x=step_x, y=step_y, mode="lines",
        line=dict(color=_ECDF1, width=1.8, shape="hv"), hoverinfo="skip",
    ))

    # D-statistic segment
    if d_stat > 0 and d_loc is not None:
        ecdf_at = np.searchsorted(xs, d_loc, side="right") / n
        cdf_at = float(dist_frozen.cdf(d_loc))
        y_bot, y_top = min(ecdf_at, cdf_at), max(ecdf_at, cdf_at)
        fig.add_trace(go.Scatter(
            x=[d_loc, d_loc], y=[y_bot, y_top], mode="lines",
            line=dict(color=_STAT, width=2.5), hoverinfo="skip",
        ))
        fig.add_annotation(
            x=d_loc, y=(y_bot + y_top) / 2,
            text=f"D\u200a=\u200a{d_stat:.4f}", showarrow=True,
            arrowhead=0, arrowcolor=_STAT, ax=45, ay=0,
            font=dict(size=10, color=_STAT),
        )

    # Legend
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        text=(f"<span style='color:{_ECDF1}'>\u2501\u2501</span> F\u2099(x)  "
              f"<span style='color:{_CDF}'>\u2508\u2508</span> {dist_label}"),
        showarrow=False, font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  KS two-sample — Fₙ₁(x) vs Fₙ₂(x)
# ─────────────────────────────────────────────────────────────────────────────
def draw_ks2_ecdf(
    sample1: np.ndarray,
    sample2: np.ndarray,
    d_stat: float,
    d_loc: float,
    alpha: float,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    xs1, ey1 = _ecdf(sample1)
    xs2, ey2 = _ecdf(sample2)
    all_min = min(xs1[0], xs2[0])
    all_max = max(xs1[-1], xs2[-1])
    pad = max((all_max - all_min) * 0.08, 0.5)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="x", font=dict(size=10, color=t["label"]))),
        yaxis=dict(**_ay, title=dict(text="F(x)", font=dict(size=10, color=t["label"])),
                   range=[-0.03, 1.07]),
    )

    step_x1 = np.concatenate([[all_min - pad], xs1, [all_max + pad]])
    step_y1 = np.concatenate([[0.0], ey1, [1.0]])
    fig.add_trace(go.Scatter(
        x=step_x1, y=step_y1, mode="lines",
        line=dict(color=_ECDF1, width=1.8, shape="hv"), hoverinfo="skip",
    ))

    step_x2 = np.concatenate([[all_min - pad], xs2, [all_max + pad]])
    step_y2 = np.concatenate([[0.0], ey2, [1.0]])
    fig.add_trace(go.Scatter(
        x=step_x2, y=step_y2, mode="lines",
        line=dict(color=_ECDF2, width=1.8, shape="hv"), hoverinfo="skip",
    ))

    # D segment
    if d_stat > 0 and d_loc is not None:
        n1, n2 = len(xs1), len(xs2)
        e1 = np.searchsorted(xs1, d_loc, side="right") / n1
        e2 = np.searchsorted(xs2, d_loc, side="right") / n2
        y_bot, y_top = min(e1, e2), max(e1, e2)
        fig.add_trace(go.Scatter(
            x=[d_loc, d_loc], y=[y_bot, y_top], mode="lines",
            line=dict(color=_STAT, width=2.5), hoverinfo="skip",
        ))
        fig.add_annotation(
            x=d_loc, y=(y_bot + y_top) / 2,
            text=f"D\u200a=\u200a{d_stat:.4f}", showarrow=True,
            arrowhead=0, arrowcolor=_STAT, ax=45, ay=0,
            font=dict(size=10, color=_STAT),
        )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        text=(f"<span style='color:{_ECDF1}'>\u2501\u2501</span> Sample\u00a01  "
              f"<span style='color:{_ECDF2}'>\u2501\u2501</span> Sample\u00a02"),
        showarrow=False, font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Chi-squared GoF — observed vs expected bars
# ─────────────────────────────────────────────────────────────────────────────
def draw_chi2_bars(
    bin_edges: np.ndarray,
    observed: np.ndarray,
    expected: np.ndarray,
    chi2_stat: float,
    df: int,
    pvalue: float,
    alpha: float,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    k = len(observed)
    edges = np.array(bin_edges, dtype=float)
    # Replace ±inf with finite bounds so centres and widths are usable
    finite = edges[np.isfinite(edges)]
    if len(finite) >= 2:
        span = finite[-1] - finite[0]
        if edges[0] == -np.inf:
            edges[0] = finite[0] - span * 0.3
        if edges[-1] == np.inf:
            edges[-1] = finite[-1] + span * 0.3
    else:
        edges = np.where(edges == -np.inf, -5.0, edges)
        edges = np.where(edges == np.inf, 5.0, edges)

    centres = (edges[:k] + edges[1:k + 1]) / 2
    width = float(np.min(np.diff(edges[:k + 1]))) * 0.38

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="Bin centre", font=dict(size=10, color=t["label"]))),
        yaxis=dict(**_ay, title=dict(text="Count", font=dict(size=10, color=t["label"]))),
    )

    # Expected
    fig.add_trace(go.Bar(
        x=centres - width / 2, y=expected, width=width,
        marker_color="rgba(148,163,184,0.45)",
        marker_line=dict(color=_EXP, width=1),
        hovertemplate="Expected: %{y:.1f}<extra></extra>",
    ))

    # Observed
    fig.add_trace(go.Bar(
        x=centres + width / 2, y=observed, width=width,
        marker_color="rgba(129,140,248,0.45)",
        marker_line=dict(color=_OBS, width=1),
        hovertemplate="Observed: %{y}<extra></extra>",
    ))

    low_bins = int(np.sum(expected < 5))
    lines = [
        f"\u03c7\u00b2\u200a=\u200a{chi2_stat:.3f}",
        f"df\u200a=\u200a{df}",
        f"p\u200a=\u200a{pvalue:.4f}",
    ]
    if low_bins > 0:
        lines.append(
            f"<span style='color:{_REJECT}'>\u26a0 E<5 in {low_bins} bin{'s' if low_bins > 1 else ''}</span>"
        )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text="<br>".join(lines), showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4, align="left",
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        text=(f"<span style='color:{_OBS}'>\u25a0</span> Observed  "
              f"<span style='color:{_EXP}'>\u25a0</span> Expected"),
        showarrow=False, font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Q-Q plot (normality)
# ─────────────────────────────────────────────────────────────────────────────
def draw_qq_plot(
    sample: np.ndarray,
    w_stat: float | None = None,
    pvalue: float | None = None,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    n = len(sample)
    sample_sorted = np.sort(sample)

    # Theoretical normal quantiles (Blom's plotting position)
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theo_q = stats.norm.ppf(probs)

    # Standardise sample
    s_mean = sample_sorted.mean()
    s_std = sample_sorted.std(ddof=1)
    if s_std < 1e-12:
        s_std = 1.0
    sample_std = (sample_sorted - s_mean) / s_std

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="Theoretical quantiles",
                                     font=dict(size=10, color=t["label"]))),
        yaxis=dict(**_ay, title=dict(text="Sample quantiles (standardised)",
                                     font=dict(size=10, color=t["label"]))),
    )

    # 95 % pointwise confidence band
    phi_sq = stats.norm.pdf(theo_q) ** 2
    phi_sq = np.maximum(phi_sq, 1e-12)
    se = np.sqrt(probs * (1 - probs) / (n * phi_sq))
    band_lo = theo_q - 1.96 * se
    band_hi = theo_q + 1.96 * se

    fig.add_trace(go.Scatter(
        x=np.concatenate([theo_q, theo_q[::-1]]),
        y=np.concatenate([band_hi, band_lo[::-1]]),
        fill="toself", fillcolor=_QQ_BAND,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
    ))

    # Reference line y = x
    margin = 0.3
    q_range = [float(theo_q[0]) - margin, float(theo_q[-1]) + margin]
    fig.add_trace(go.Scatter(
        x=q_range, y=q_range, mode="lines",
        line=dict(color=_QQ_REF, width=1.2, dash="dash"), hoverinfo="skip",
    ))

    # Q-Q points
    fig.add_trace(go.Scatter(
        x=theo_q, y=sample_std, mode="markers",
        marker=dict(color=_QQ_PT, size=5, opacity=0.7, line=dict(width=0)),
        hovertemplate="Theo: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
    ))

    if w_stat is not None:
        lines = [f"W\u200a=\u200a{w_stat:.4f}"]
        if pvalue is not None:
            lines.append(f"p\u200a=\u200a{pvalue:.4f}")
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
# 5.  Null distribution + test statistic
# ─────────────────────────────────────────────────────────────────────────────
def draw_gof_null_dist(
    stat_value: float | None,
    test_type: str,
    null_params: dict,
    alpha: float,
    alternative: str = "two-sided",
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # ── Shapiro–Wilk: bootstrap null ─────────────────────────────────────────
    if test_type == "sw":
        w_null = null_params.get("w_null", np.array([]))
        if len(w_null) < 10:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="Draw a sample to see the null distribution",
                showarrow=False, font=dict(size=12, color=t["muted"]),
            )
            return fig

        fig.update_layout(xaxis=dict(
            **_ax, title=dict(text="W (Shapiro\u2013Wilk)",
                              font=dict(size=10, color=t["label"])),
        ))

        fig.add_trace(go.Histogram(
            x=w_null, nbinsx=40,
            marker_color="rgba(148,163,184,0.3)",
            marker_line=dict(color=_CDF, width=0.5),
            histnorm="probability density", hoverinfo="skip",
        ))

        try:
            kde = gaussian_kde(w_null)
            xg = np.linspace(float(w_null.min()), float(w_null.max()), 200)
            kde_y = kde(xg)
            fig.add_trace(go.Scatter(
                x=xg, y=kde_y, mode="lines",
                line=dict(color=_CDF, width=1.5), hoverinfo="skip",
            ))
            # Left-tail rejection region
            cv = float(np.percentile(w_null, alpha * 100))
            left = xg <= cv
            if left.any():
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xg[left], xg[left][::-1]]),
                    y=np.concatenate([kde_y[left], np.zeros(left.sum())]),
                    fill="toself", fillcolor="rgba(248,113,113,0.30)",
                    line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                ))
            fig.add_vline(x=cv, line_dash="dot", line_color=_REJECT, line_width=0.9)
        except Exception:
            pass

        if stat_value is not None:
            fig.add_vline(x=stat_value, line_dash="solid", line_color=_STAT, line_width=2)
            fig.add_annotation(
                x=stat_value, y=1, yref="paper", yshift=-12,
                text=f"W\u200a=\u200a{stat_value:.4f}", showarrow=False,
                font=dict(size=10, color=_STAT),
            )
        return fig

    # ── KS / Chi-squared: parametric null ────────────────────────────────────
    if test_type in ("ks1", "ks2"):
        n = null_params.get("n", 30)
        n2 = null_params.get("n2", n)
        n_eff = n if test_type == "ks1" else int(n * n2 / (n + n2))
        stat_label = "D"

        one_sided = (alternative != "two-sided") and (test_type == "ks1")
        try:
            ndist = stats.ksone(n_eff) if one_sided else stats.kstwo(n_eff)
            x_hi = float(ndist.ppf(0.995))
            xs = np.linspace(0.002, x_hi, 300)
            ys = ndist.pdf(xs)
            cv = float(ndist.ppf(1 - alpha))
        except Exception:
            # Fallback: asymptotic Kolmogorov
            sq = np.sqrt(n_eff)
            x_hi = float(stats.kstwobign.ppf(0.999)) / sq
            xs = np.linspace(0.002, x_hi, 300)
            ys = sq * stats.kstwobign.pdf(sq * xs)
            cv = float(stats.kstwobign.ppf(1 - alpha)) / sq

        fig.update_layout(xaxis=dict(
            **_ax, title=dict(text=stat_label, font=dict(size=10, color=t["label"])),
        ))

    elif test_type == "chi2":
        df = max(null_params.get("df", 1), 1)
        ndist = stats.chi2(df)
        x_hi = float(ndist.ppf(0.999))
        xs = np.linspace(0.01, x_hi, 300)
        ys = ndist.pdf(xs)
        cv = float(ndist.ppf(1 - alpha))
        stat_label = "\u03c7\u00b2"

        fig.update_layout(xaxis=dict(
            **_ax, title=dict(text=f"{stat_label} (df\u200a=\u200a{df})",
                              font=dict(size=10, color=t["label"])),
        ))
    else:
        return fig

    # Null PDF
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_CDF, width=1.5), hoverinfo="skip",
    ))

    # Rejection region (right tail)
    rej = xs >= cv
    if rej.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs[rej], xs[rej][::-1]]),
            y=np.concatenate([ys[rej], np.zeros(rej.sum())]),
            fill="toself", fillcolor="rgba(248,113,113,0.30)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        ))

    fig.add_vline(x=cv, line_dash="dot", line_color=_REJECT, line_width=0.9)

    if stat_value is not None:
        fig.add_vline(x=stat_value, line_dash="solid", line_color=_STAT, line_width=2)
        fig.add_annotation(
            x=stat_value, y=1, yref="paper", yshift=-12,
            text=f"{stat_label}\u200a=\u200a{stat_value:.4f}", showarrow=False,
            font=dict(size=10, color=_STAT),
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  p-value accumulation histogram
# ─────────────────────────────────────────────────────────────────────────────
def draw_gof_pvalue_hist(
    pvalues: list | np.ndarray,
    alpha: float,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="p-value",
                                     font=dict(size=10, color=t["label"])),
                   range=[0, 1]),
        yaxis=dict(**_ay, title=dict(text="Count",
                                     font=dict(size=10, color=t["label"]))),
    )

    if not len(pvalues):
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Draw samples to see p-value distribution",
            showarrow=False, font=dict(size=12, color=t["muted"]),
        )
        return fig

    pv = np.asarray(pvalues)
    fig.add_trace(go.Histogram(
        x=pv, xbins=dict(start=0, end=1, size=0.05),
        marker_color="rgba(129,140,248,0.45)",
        marker_line=dict(color=_ECDF1, width=0.5),
        hovertemplate="Bin: %{x}<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(x=alpha, line_dash="dash", line_color=_REJECT, line_width=1.2)
    fig.add_annotation(
        x=alpha, y=1, yref="paper", yshift=-5,
        text=f"\u03b1\u200a=\u200a{alpha}", showarrow=False,
        font=dict(size=9, color=_REJECT), xanchor="left", xshift=3,
    )

    rej_rate = float(np.mean(pv < alpha))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text=f"Reject rate\u200a=\u200a{rej_rate:.3f}",
        showarrow=False, font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"],
        borderwidth=1, borderpad=4,
    )
    return fig
