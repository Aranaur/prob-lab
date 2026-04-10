# =============================================================================
# p-value Explorer — Plotly chart builders
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import nct as nct_dist
import plotly.graph_objects as go

from plots import _DARK_LAYOUT, _LIGHT_LAYOUT, _LABEL, _LIGHT_LABEL, _base_fig, _theme

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Null distribution with rejection region + p-value area
#     method="t" → t(df),   method="z" → N(0,1)
# ─────────────────────────────────────────────────────────────────────────────
def draw_null_dist_plot(
    last_stat: float | None,
    df: int,
    alpha: float,
    alternative: str,
    method: str = "t",
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    is_z = method == "z"
    dist  = stats.norm if is_z else stats.t
    d_kw  = {} if is_z else {"df": df}
    x_lab = "z-statistic" if is_z else "t-statistic"
    s_lab = "z" if is_z else "t"

    x_lim = max(5.0, abs(last_stat) * 1.35) if last_stat is not None else 5.0
    xs = np.linspace(-x_lim, x_lim, 600)
    ys = dist.pdf(xs, **d_kw)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text=x_lab, font=dict(size=10, color=t["label"])),
            range=[-x_lim, x_lim],
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # Full distribution curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="#94a3b8", width=1.5),
        hoverinfo="skip",
    ))

    # Critical values and rejection region shading
    if alternative == "two-sided":
        cv = dist.ppf(1 - alpha / 2, **d_kw)
        crit_regions = [("left", -cv), ("right", cv)]
    elif alternative == "greater":
        cv = dist.ppf(1 - alpha, **d_kw)
        crit_regions = [("right", cv)]
    else:
        cv = dist.ppf(alpha, **d_kw)
        crit_regions = [("left", cv)]

    for side, c in crit_regions:
        mask = xs <= c if side == "left" else xs >= c
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs[mask], xs[mask][::-1]]),
            y=np.concatenate([ys[mask], np.zeros(mask.sum())]),
            fill="toself", fillcolor="rgba(248,113,113,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        ))
        fig.add_vline(x=c, line_dash="dot", line_color="#f87171", line_width=0.9)

    if last_stat is None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Press Sample to begin", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    # p-value and p-value shading mask
    if alternative == "two-sided":
        pvalue  = float(2 * dist.cdf(-abs(last_stat), **d_kw))
        pv_mask = (xs <= -abs(last_stat)) | (xs >= abs(last_stat))
    elif alternative == "greater":
        pvalue  = float(1 - dist.cdf(last_stat, **d_kw))
        pv_mask = xs >= last_stat
    else:
        pvalue  = float(dist.cdf(last_stat, **d_kw))
        pv_mask = xs <= last_stat

    rejected = pvalue < alpha
    pv_fill  = "rgba(248,113,113,0.55)" if rejected else "rgba(56,189,248,0.40)"
    line_col = "#f87171" if rejected else "#38bdf8"

    if alternative == "two-sided":
        pv_masks = [xs <= -abs(last_stat), xs >= abs(last_stat)]
    else:
        pv_masks = [pv_mask]

    for m in pv_masks:
        if m.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([xs[m], xs[m][::-1]]),
                y=np.concatenate([ys[m], np.zeros(m.sum())]),
                fill="toself", fillcolor=pv_fill,
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            ))

    fig.add_vline(x=last_stat, line_color=line_col, line_width=2)

    p_str    = f"{pvalue:.4f}" if pvalue >= 0.0001 else "<0.0001"
    decision = "Reject H\u2080" if rejected else "Fail to reject H\u2080"
    dist_lab = "N(0,\u00a01)" if is_z else f"t({df})"
    muted_col = t["muted"]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text=(
            f"{s_lab} = {last_stat:.3f}<br>"
            f"p = {p_str}<br>"
            f"<b>{decision}</b><br>"
            f"<span style='color:{muted_col}'>{dist_lab}</span>"
        ),
        showarrow=False,
        font=dict(size=10, color=line_col),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=4,
        align="right",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  p-value histogram (accumulated over many tests)
# ─────────────────────────────────────────────────────────────────────────────
def draw_pvalue_hist(
    pvalues: list,
    alpha: float,
    dark: bool = True,
    pvalues_wilcoxon: list | None = None,
    wilcoxon_label: str = "Wilcoxon",
    param_label: str = "t-test",
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    dual = pvalues_wilcoxon is not None and len(pvalues_wilcoxon) >= 5

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="p-value", font=dict(size=10, color=t["label"])),
            range=[0, 1],
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if len(pvalues) < 5:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    # Parametric histogram
    fig.add_trace(go.Histogram(
        x=pvalues,
        xbins=dict(start=0, end=1, size=0.05),
        histnorm="probability density",
        marker=dict(
            color="rgba(56,189,248,0.35)",
            line=dict(color="#38bdf8", width=0.6),
        ),
        name=param_label if dual else None,
        showlegend=dual,
        hovertemplate="p \u2208 [%{x:.2f}, %{x:.2f}+0.05)<br>Density=%{y:.2f}<extra></extra>",
    ))

    # Wilcoxon overlay
    if dual:
        fig.add_trace(go.Histogram(
            x=pvalues_wilcoxon,
            xbins=dict(start=0, end=1, size=0.05),
            histnorm="probability density",
            marker=dict(
                color="rgba(249,115,22,0.35)",
                line=dict(color="#f97316", width=0.6),
            ),
            name=wilcoxon_label,
            showlegend=True,
            hovertemplate="p \u2208 [%{x:.2f}, %{x:.2f}+0.05)<br>Density=%{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            barmode="overlay",
            legend=dict(
                font=dict(size=9, color=t["annot_text"]),
                bgcolor="rgba(0,0,0,0)",
                x=0.98, y=0.60, xanchor="right", yanchor="top",
            ),
        )

    fig.add_vline(x=alpha, line_dash="dash", line_color="#f87171", line_width=1.2)
    fig.add_annotation(
        x=alpha, y=1, yref="paper",
        text=f"\u03b1\u200a=\u200a{alpha}",
        showarrow=False,
        font=dict(size=10, color="#f87171"),
        xanchor="left", yanchor="top", xshift=5,
    )

    reject_frac = sum(p < alpha for p in pvalues) / len(pvalues)
    if dual:
        wil_reject = sum(p < alpha for p in pvalues_wilcoxon) / len(pvalues_wilcoxon)
        annot_text = (
            f"<span style='color:#38bdf8'>{param_label}: {reject_frac:.1%}</span><br>"
            f"<span style='color:#f97316'>{wilcoxon_label}: {wil_reject:.1%}</span>"
        )
    else:
        annot_text = f"Reject rate: {reject_frac:.1%}"

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text=annot_text,
        showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Power diagram: H₀ and H₁ distributions with α / β / power regions
# ─────────────────────────────────────────────────────────────────────────────
def draw_power_diagram(
    mu0: float,
    mu_true: float,
    se_val: float,
    df: int,
    alpha: float,
    alternative: str,
    empirical_rate: float | None = None,
    method: str = "t",
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    se     = max(se_val, 1e-9)
    effect = mu_true - mu0

    def _cv(p: float) -> float:
        if method == "z":
            return float(stats.norm.ppf(p))
        return float(stats.t.ppf(p, max(df, 1)))

    spread = max(abs(effect) + 4.5 * se, 5.0 * se)
    center = (mu0 + mu_true) / 2
    xs = np.linspace(center - spread, center + spread, 500)

    y0 = stats.norm.pdf(xs, mu0, se)
    y1 = stats.norm.pdf(xs, mu_true, se)

    if method == "z":
        if alternative == "two-sided":
            cv_lo = mu0 + _cv(alpha / 2)       * se
            cv_hi = mu0 + _cv(1 - alpha / 2)   * se
            theo_power  = (stats.norm.cdf(cv_lo, mu_true, se) +
                           1 - stats.norm.cdf(cv_hi, mu_true, se))
            alpha_masks = [xs <= cv_lo, xs >= cv_hi]
            power_masks = [xs <= cv_lo, xs >= cv_hi]
            cv_lines    = [cv_lo, cv_hi]
        elif alternative == "greater":
            cv_hi       = mu0 + _cv(1 - alpha) * se
            theo_power  = 1 - stats.norm.cdf(cv_hi, mu_true, se)
            alpha_masks = [xs >= cv_hi]
            power_masks = [xs >= cv_hi]
            cv_lines    = [cv_hi]
        else:
            cv_lo       = mu0 + _cv(alpha)     * se
            theo_power  = stats.norm.cdf(cv_lo, mu_true, se)
            alpha_masks = [xs <= cv_lo]
            power_masks = [xs <= cv_lo]
            cv_lines    = [cv_lo]
    else:
        ncp = (mu_true - mu0) / se
        if alternative == "two-sided":
            cv_lo = mu0 + _cv(alpha / 2)       * se
            cv_hi = mu0 + _cv(1 - alpha / 2)   * se
            tc = stats.t.ppf(1 - alpha / 2, max(df,1))
            tp = nct_dist.cdf(-tc, max(df,1), ncp) + 1 - nct_dist.cdf(tc, max(df,1), ncp)
            theo_power = 1.0 if np.isnan(tp) else tp
            alpha_masks = [xs <= cv_lo, xs >= cv_hi]
            power_masks = [xs <= cv_lo, xs >= cv_hi]
            cv_lines    = [cv_lo, cv_hi]
        elif alternative == "greater":
            cv_hi       = mu0 + _cv(1 - alpha) * se
            tc = stats.t.ppf(1 - alpha, max(df,1))
            tp = 1 - nct_dist.cdf(tc, max(df,1), ncp)
            theo_power = 1.0 if np.isnan(tp) else tp
            alpha_masks = [xs >= cv_hi]
            power_masks = [xs >= cv_hi]
            cv_lines    = [cv_hi]
        else:
            cv_lo       = mu0 + _cv(alpha)     * se
            tc = stats.t.ppf(1 - alpha, max(df,1))
            tp = nct_dist.cdf(-tc, max(df,1), ncp)
            theo_power = 1.0 if np.isnan(tp) else tp
            alpha_masks = [xs <= cv_lo]
            power_masks = [xs <= cv_lo]
            cv_lines    = [cv_lo]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            **_ax,
            title=dict(text="Sample mean (x\u0304)", font=dict(size=10, color=t["label"])),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    fig.add_trace(go.Scatter(
        x=xs, y=y0, mode="lines",
        line=dict(color="#94a3b8", width=1.4),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=y1, mode="lines",
        line=dict(color="#c084fc", width=1.4),
        hoverinfo="skip",
    ))

    for mask in alpha_masks:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([xs[mask], xs[mask][::-1]]),
                y=np.concatenate([y0[mask], np.zeros(mask.sum())]),
                fill="toself", fillcolor="rgba(248,113,113,0.35)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            ))

    for mask in power_masks:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([xs[mask], xs[mask][::-1]]),
                y=np.concatenate([y1[mask], np.zeros(mask.sum())]),
                fill="toself", fillcolor="rgba(192,132,252,0.45)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            ))

    for cv in cv_lines:
        fig.add_vline(x=cv, line_dash="dot", line_color="#f87171", line_width=0.9)

    fig.add_vline(x=mu0,    line_dash="dash", line_color="#94a3b8", line_width=1)
    if abs(effect) > 1e-9:
        fig.add_vline(x=mu_true, line_dash="dash", line_color="#c084fc", line_width=1)

    delta     = effect / se if se > 0 else 0.0
    cv_label  = "t-crit" if method == "t" else "z-crit"
    muted_col = t["muted"]
    lines = [
        f"\u03b4 (Cohen\u2019s d) = {delta:.3f}",
        f"Theoretical power\u200a=\u200a{theo_power:.3f}",
        f"\u03b2 (Type\u00a0II)\u200a=\u200a{1 - theo_power:.3f}",
        f"<span style='color:{muted_col}'>Critical values: {cv_label}</span>",
    ]
    if empirical_rate is not None:
        label = "Empirical power" if abs(effect) > 1e-9 else "Type\u00a0I rate"
        lines.insert(3, f"{label}\u200a=\u200a{empirical_rate:.3f}")

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text="<br>".join(lines),
        showarrow=False,
        font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border2"], borderwidth=1, borderpad=4,
        align="left",
    )

    return fig
