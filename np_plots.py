# =============================================================================
# Nonparametric Explorer — Plotly chart builders
# =============================================================================

import numpy as np
from scipy.stats import gaussian_kde, rankdata
import plotly.graph_objects as go

from plots import _DARK_LAYOUT, _LIGHT_LAYOUT, _base_fig, _theme

# Colour tokens
_C_PARAM    = "#38bdf8"   # cyan  — parametric test
_C_NONPARAM = "#f97316"   # orange — nonparametric test
_C_GROUP_A  = "#38bdf8"   # cyan
_C_GROUP_B  = "#c084fc"   # purple
_C_POS      = "#34d399"   # green  — positive differences
_C_NEG      = "#f87171"   # red    — negative differences


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sample data KDE + rug
#     Independent (sample_b given): two KDEs.
#     Paired      (sample_b None):  one KDE of differences with zero-line.
# ─────────────────────────────────────────────────────────────────────────────
def draw_np_sample_kde(
    sample_a: np.ndarray | None,
    sample_b: np.ndarray | None = None,
    label_a: str = "Group A",
    label_b: str = "Group B",
    prob_a_gt_b: float | None = None,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="Value", font=dict(size=10, color=t["label"]))),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if sample_a is None or len(sample_a) < 2:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Press Sample to begin", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    # ── Independent mode ─────────────────────────────────────────────────
    if sample_b is not None and len(sample_b) >= 2:
        all_v = np.concatenate([sample_a, sample_b])
        lo, hi = float(all_v.min()), float(all_v.max())
        pad = max((hi - lo) * 0.15, 0.5)
        xs = np.linspace(lo - pad, hi + pad, 300)

        for samp, col in [(sample_a, _C_GROUP_A), (sample_b, _C_GROUP_B)]:
            try:
                kde = gaussian_kde(samp)
                fig.add_trace(go.Scatter(
                    x=xs, y=kde(xs), mode="lines",
                    line=dict(color=col, width=1.8), hoverinfo="skip",
                ))
            except Exception:
                pass

        for samp, col, ry, lab in [
            (sample_a, _C_GROUP_A, -0.015, label_a),
            (sample_b, _C_GROUP_B, -0.04,  label_b),
        ]:
            fig.add_trace(go.Scatter(
                x=samp, y=np.full(len(samp), ry), mode="markers",
                marker=dict(symbol="line-ns", size=8, color=col, opacity=0.7,
                            line=dict(color=col, width=1)),
                hovertemplate=f"{lab}: %{{x:.3f}}<extra></extra>",
            ))

        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            text=(f'<span style="color:{_C_GROUP_A}">\u2501</span> {label_a}'
                  f'<br><span style="color:{_C_GROUP_B}">\u2501</span> {label_b}'),
            showarrow=False, font=dict(size=10, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        )

        if prob_a_gt_b is not None:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.98, y=0.98,
                xanchor="right", yanchor="top",
                text=f"P\u0302(A\u2009>\u2009B) = {prob_a_gt_b:.3f}",
                showarrow=False, font=dict(size=10, color=t["annot_text"]),
                bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
            )

    # ── Paired mode (differences) ────────────────────────────────────────
    else:
        diffs = sample_a
        lo, hi = float(diffs.min()), float(diffs.max())
        pad = max((hi - lo) * 0.15, 0.5)
        xs = np.linspace(lo - pad, hi + pad, 300)

        try:
            kde = gaussian_kde(diffs)
            fig.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines",
                line=dict(color=_C_PARAM, width=1.8), hoverinfo="skip",
            ))
        except Exception:
            pass

        fig.add_trace(go.Scatter(
            x=diffs, y=np.full(len(diffs), -0.015), mode="markers",
            marker=dict(symbol="line-ns", size=8, color=_C_PARAM, opacity=0.7,
                        line=dict(color=_C_PARAM, width=1)),
            hovertemplate="d = %{x:.3f}<extra></extra>",
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="#f87171", line_width=1)

        mean_d = float(np.mean(diffs))
        med_d  = float(np.median(diffs))
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            text=f"d\u0304 = {mean_d:.3f}<br>median = {med_d:.3f}",
            showarrow=False, font=dict(size=10, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        )
        fig.update_xaxes(title=dict(text="Difference (d)", font=dict(size=10, color=t["label"])))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dual p-value histogram (parametric vs nonparametric)
# ─────────────────────────────────────────────────────────────────────────────
def draw_np_pvalue_hist(
    pvals_param: list,
    pvals_nonparam: list,
    alpha: float,
    param_label: str = "t-test",
    nonparam_label: str = "Mann-Whitney U",
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="p-value", font=dict(size=10, color=t["label"])), range=[0, 1]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if len(pvals_param) < 5:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    fig.add_trace(go.Histogram(
        x=pvals_param,
        xbins=dict(start=0, end=1, size=0.05),
        histnorm="probability density",
        marker=dict(color="rgba(56,189,248,0.25)", line=dict(color=_C_PARAM, width=0.6)),
        hovertemplate=(f"{param_label}<br>"
                       "p \u2208 [%{x:.2f}, +0.05)<br>Density=%{y:.2f}<extra></extra>"),
    ))
    fig.add_trace(go.Histogram(
        x=pvals_nonparam,
        xbins=dict(start=0, end=1, size=0.05),
        histnorm="probability density",
        marker=dict(color="rgba(249,115,22,0.25)", line=dict(color=_C_NONPARAM, width=0.6)),
        hovertemplate=(f"{nonparam_label}<br>"
                       "p \u2208 [%{x:.2f}, +0.05)<br>Density=%{y:.2f}<extra></extra>"),
    ))
    fig.update_layout(barmode="overlay")

    fig.add_vline(x=alpha, line_dash="dash", line_color="#f87171", line_width=1.2)
    fig.add_annotation(
        x=alpha, y=1, yref="paper",
        text=f"\u03b1\u200a=\u200a{alpha}", showarrow=False,
        font=dict(size=10, color="#f87171"),
        xanchor="left", yanchor="top", xshift=5,
    )

    n_p  = len(pvals_param)
    n_np = len(pvals_nonparam)
    pr  = sum(p < alpha for p in pvals_param) / n_p if n_p else 0
    nr  = sum(p < alpha for p in pvals_nonparam) / n_np if n_np else 0

    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text=(f'<span style="color:{_C_PARAM}">{param_label}: {pr:.1%}</span><br>'
              f'<span style="color:{_C_NONPARAM}">{nonparam_label}: {nr:.1%}</span>'),
        showarrow=False, font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        align="right",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        text=(f'<span style="color:{_C_PARAM}">\u2588</span> {param_label}'
              f'<br><span style="color:{_C_NONPARAM}">\u2588</span> {nonparam_label}'),
        showarrow=False, font=dict(size=10, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Reject-rate horizontal bar chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_np_reject_bars(
    n_total: int,
    n_param_rej: int,
    n_nonparam_rej: int,
    alpha: float,
    param_label: str = "t-test",
    nonparam_label: str = "Mann-Whitney U",
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            range=[0, 1], tickformat=".0%",
            gridcolor=t["grid"], gridwidth=0.3, linecolor=t["grid"],
            tickfont=dict(size=10, color=t["axis"]),
            title=dict(text="Reject rate", font=dict(size=10, color=t["label"])),
            zeroline=False,
        ),
        yaxis=dict(showticklabels=True, showgrid=False, zeroline=False,
                   tickfont=dict(size=9, color=t["axis"])),
        margin=dict(l=100, r=12, t=8, b=36),
    )

    if n_total == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    pr = n_param_rej / n_total
    nr = n_nonparam_rej / n_total

    fig.add_trace(go.Bar(
        y=[nonparam_label, param_label],
        x=[nr, pr],
        orientation="h",
        marker=dict(color=[_C_NONPARAM, _C_PARAM]),
        text=[f"{nr:.1%}", f"{pr:.1%}"],
        textposition="inside",
        textfont=dict(color="#fff", size=11, family="Inter"),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))

    fig.add_vline(x=alpha, line_dash="dash", line_color="#f87171", line_width=1.2)
    fig.add_annotation(
        x=alpha, y=1.05, yref="paper",
        text=f"\u03b1 = {alpha}", showarrow=False,
        font=dict(size=9, color="#f87171"),
        xanchor="left", yanchor="bottom", xshift=3,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Rank visualisation
#     Independent: combined sorted sample coloured by group.
#     Paired:      differences sorted by |d|, coloured by sign.
# ─────────────────────────────────────────────────────────────────────────────
def draw_np_rank_plot(
    sample_a: np.ndarray | None,
    sample_b: np.ndarray | None = None,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]
    _ay = _DARK_LAYOUT["yaxis"] if dark else _LIGHT_LAYOUT["yaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(text="Rank", font=dict(size=10, color=t["label"]))),
        yaxis=dict(**_ay, title=dict(text="Value", font=dict(size=10, color=t["label"]))),
    )

    if sample_a is None or len(sample_a) < 2:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Press Sample to begin", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    # ── Independent: MW-U ranks ──────────────────────────────────────────
    if sample_b is not None and len(sample_b) >= 2:
        na, nb = len(sample_a), len(sample_b)
        combined = np.concatenate([sample_a, sample_b])
        groups = np.array(["A"] * na + ["B"] * nb)
        order = np.argsort(combined)
        ranks = np.arange(1, len(combined) + 1)
        sorted_vals = combined[order]
        sorted_grp  = groups[order]

        for g, col, lab in [("A", _C_GROUP_A, "Group A"), ("B", _C_GROUP_B, "Group B")]:
            mask = sorted_grp == g
            fig.add_trace(go.Scatter(
                x=ranks[mask], y=sorted_vals[mask], mode="markers",
                marker=dict(color=col, size=7, opacity=0.85),
                hovertemplate=f"Rank %{{x}}<br>Value: %{{y:.3f}}<extra>{lab}</extra>",
            ))

        rs_a = float(ranks[sorted_grp == "A"].sum())
        rs_b = float(ranks[sorted_grp == "B"].sum())
        u_a  = rs_a - na * (na + 1) / 2
        u_b  = rs_b - nb * (nb + 1) / 2

        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            text=(f'<span style="color:{_C_GROUP_A}">R\u2090={rs_a:.0f}  U\u2090={u_a:.0f}</span><br>'
                  f'<span style="color:{_C_GROUP_B}">R\u1d47={rs_b:.0f}  U\u1d47={u_b:.0f}</span>'),
            showarrow=False, font=dict(size=10, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        )

        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            text=(f'<span style="color:{_C_GROUP_A}">\u25cf</span> Group A'
                  f'<br><span style="color:{_C_GROUP_B}">\u25cf</span> Group B'),
            showarrow=False, font=dict(size=10, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        )

    # ── Paired: Wilcoxon signed-ranks ────────────────────────────────────
    else:
        diffs = sample_a
        nz_mask = np.abs(diffs) > 1e-12
        nz = diffs[nz_mask]

        if len(nz) < 2:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="Not enough non-zero differences", showarrow=False,
                font=dict(size=13, color=t["muted"]),
            )
            return fig

        abs_r = rankdata(np.abs(nz))
        signs = np.sign(nz)
        order = np.argsort(abs_r)

        for sv, col, lab in [(1, _C_POS, "d > 0"), (-1, _C_NEG, "d < 0")]:
            mask = signs[order] == sv
            fig.add_trace(go.Scatter(
                x=abs_r[order][mask], y=nz[order][mask], mode="markers",
                marker=dict(color=col, size=7, opacity=0.85),
                hovertemplate=f"Rank %{{x}}<br>d = %{{y:.3f}}<extra>{lab}</extra>",
            ))

        w_plus  = float(np.sum(abs_r[signs > 0]))
        w_minus = float(np.sum(abs_r[signs < 0]))

        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            text=(f'<span style="color:{_C_POS}">W\u207a = {w_plus:.0f}</span><br>'
                  f'<span style="color:{_C_NEG}">W\u207b = {w_minus:.0f}</span>'),
            showarrow=False, font=dict(size=10, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        )

        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            text=(f'<span style="color:{_C_POS}">\u25cf</span> d > 0'
                  f'<br><span style="color:{_C_NEG}">\u25cf</span> d < 0'),
            showarrow=False, font=dict(size=10, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
        )

        fig.update_xaxes(title=dict(text="Rank of |d|", font=dict(size=10, color=t["label"])))
        fig.update_yaxes(title=dict(text="Difference (d)", font=dict(size=10, color=t["label"])))

    return fig
