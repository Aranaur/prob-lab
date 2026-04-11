# =============================================================================
# Multiple Testing Explorer — Plotly chart builders
# =============================================================================

import numpy as np
import plotly.graph_objects as go

from plots import _DARK_LAYOUT, _LIGHT_LAYOUT, _base_fig, _theme

# ── Color tokens ─────────────────────────────────────────────────────────────
_C_H0   = "#94a3b8"   # gray  – null hypotheses
_C_H1   = "#34d399"   # green – true alternatives / true positives
_C_FP   = "#f87171"   # red   – false positives
_C_BONF = "#fbbf24"   # yellow
_C_HOLM = "#f97316"   # orange
_C_BH   = "#38bdf8"   # cyan
_C_BY   = "#c084fc"   # purple

_METHOD_COLORS = {
    "none": _C_FP, "bonferroni": _C_BONF,
    "holm": _C_HOLM, "bh": _C_BH, "by": _C_BY,
}
_METHOD_LABELS = {
    "none": "None", "bonferroni": "Bonferroni",
    "holm": "Holm", "bh": "BH", "by": "BY",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  p-value scatter (rank or original index)
# ─────────────────────────────────────────────────────────────────────────────
def draw_mt_pvalue_scatter(
    pvalues: np.ndarray | None,
    is_h1: np.ndarray | None,
    alpha: float,
    sort_by_rank: bool = True,
    file_drawer: bool = False,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(
            text="Rank (sorted by p)" if sort_by_rank else "Test index",
            font=dict(size=10, color=t["label"]),
        )),
        yaxis=dict(
            showticklabels=True, showgrid=True,
            gridcolor=t["grid"], gridwidth=0.3,
            tickfont=dict(size=9, color=t["axis"]),
            title=dict(text="p-value", font=dict(size=10, color=t["label"])),
            zeroline=False,
        ),
    )

    if pvalues is None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Press Sample to begin", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    m = len(pvalues)

    if sort_by_rank:
        order = np.argsort(pvalues)
        xs = np.arange(1, m + 1)
        pv = pvalues[order]
        h1 = is_h1[order]
    else:
        xs = np.arange(1, m + 1)
        pv = pvalues
        h1 = is_h1

    if file_drawer:
        keep = pv < 0.05
        if keep.sum() == 0:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="No p < 0.05 in this experiment", showarrow=False,
                font=dict(size=12, color=t["muted"]),
            )
            return fig
        xs = np.arange(1, keep.sum() + 1) if sort_by_rank else xs[keep]
        pv = pv[keep]
        h1 = h1[keep]

    # H₀ dots
    h0m = ~h1
    if h0m.any():
        fig.add_trace(go.Scatter(
            x=xs[h0m], y=pv[h0m], mode="markers",
            marker=dict(color=_C_H0, size=6, opacity=0.8),
            name="H\u2080 (null)", showlegend=True,
            hovertemplate="Test %{x}<br>p=%{y:.4f}<extra>H\u2080</extra>",
        ))
    if h1.any():
        fig.add_trace(go.Scatter(
            x=xs[h1], y=pv[h1], mode="markers",
            marker=dict(color=_C_H1, size=7, symbol="diamond", opacity=0.9),
            name="H\u2081 (effect)", showlegend=True,
            hovertemplate="Test %{x}<br>p=%{y:.4f}<extra>H\u2081</extra>",
        ))

    # Threshold lines
    if not file_drawer:
        fig.add_hline(y=alpha, line_dash="dash", line_color=_C_FP, line_width=1)
        fig.add_annotation(
            x=1, xref="paper", y=alpha, yref="y",
            text=f"\u03b1={alpha}", showarrow=False,
            font=dict(size=8, color=_C_FP), xanchor="right", yshift=8,
        )

        if sort_by_rank:
            ranks_full = np.arange(1, m + 1)
            # BH diagonal
            fig.add_trace(go.Scatter(
                x=ranks_full, y=ranks_full / m * alpha, mode="lines",
                line=dict(color=_C_BH, width=1.2, dash="dash"),
                name="BH", showlegend=True, hoverinfo="skip",
            ))
            # BY diagonal
            c_m = float(np.sum(1.0 / ranks_full))
            fig.add_trace(go.Scatter(
                x=ranks_full, y=ranks_full / (m * c_m) * alpha, mode="lines",
                line=dict(color=_C_BY, width=1.2, dash="dash"),
                name="BY", showlegend=True, hoverinfo="skip",
            ))
            # Bonferroni
            fig.add_hline(y=alpha / m, line_dash="dot", line_color=_C_BONF,
                          line_width=0.8)
        else:
            fig.add_hline(y=alpha / m, line_dash="dot", line_color=_C_BONF,
                          line_width=0.8)
            fig.add_annotation(
                x=1, xref="paper", y=alpha / m, yref="y",
                text=f"\u03b1/m={alpha/m:.4f}", showarrow=False,
                font=dict(size=8, color=_C_BONF), xanchor="right", yshift=-8,
            )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(size=8, color=t["annot_text"]),
            bgcolor="rgba(0,0,0,0)",
            x=0.98, y=0.02, xanchor="right", yanchor="bottom",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Correction comparison — stacked bars (TP + FP = Discoveries)
# ─────────────────────────────────────────────────────────────────────────────
def draw_mt_correction_bars(
    method_stats: dict,
    total: int,
    k: int,
    alpha: float,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)

    fig = _base_fig(
        dark=dark,
        xaxis=dict(
            tickfont=dict(size=9, color=t["axis"]),
            showgrid=False,
        ),
        yaxis=dict(
            showticklabels=True, showgrid=True,
            gridcolor=t["grid"], gridwidth=0.3,
            tickfont=dict(size=9, color=t["axis"]),
            title=dict(text="Avg. discoveries / experiment",
                       font=dict(size=10, color=t["label"])),
            zeroline=False, rangemode="tozero",
        ),
    )

    if total == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    methods = ["none", "bonferroni", "holm", "bh", "by"]
    labels = [_METHOD_LABELS[m] for m in methods]
    tp_vals = [method_stats[m]["tp"] / total for m in methods]
    fp_vals = [method_stats[m]["fp"] / total for m in methods]

    if k > 0:
        fig.add_trace(go.Bar(
            x=labels, y=tp_vals, name="True Positives",
            marker_color=_C_H1, opacity=0.85, showlegend=True,
            hovertemplate="%{x}<br>TP=%{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Bar(
        x=labels, y=fp_vals, name="False Positives",
        marker_color=_C_FP, opacity=0.85, showlegend=True,
        hovertemplate="%{x}<br>FP=%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        showlegend=True,
        legend=dict(
            font=dict(size=8, color=t["annot_text"]),
            bgcolor="rgba(0,0,0,0)",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
        ),
    )

    # α reference line (expected FP per method = m₀ · α for uncorrected)
    fig.add_hline(y=alpha, line_dash="dot", line_color=_C_FP, line_width=0.8,
                  annotation_text=f"\u03b1={alpha}", annotation_position="top right",
                  annotation_font_size=8, annotation_font_color=_C_FP)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FWER accumulation curve — theoretical + empirical
# ─────────────────────────────────────────────────────────────────────────────
def draw_mt_fwer_curve(
    m: int,
    alpha: float,
    empirical_fwer: float | None,
    total: int,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(
            text="Number of hypotheses (m)",
            font=dict(size=10, color=t["label"]),
        )),
        yaxis=dict(
            showticklabels=True, showgrid=True,
            gridcolor=t["grid"], gridwidth=0.3,
            tickfont=dict(size=9, color=t["axis"]),
            title=dict(text="P(\u2265 1 false positive)",
                       font=dict(size=10, color=t["label"])),
            range=[0, 1.05], zeroline=False,
        ),
    )

    ms = np.arange(1, max(m, 2) + 1)
    theoretical = 1 - (1 - alpha) ** ms

    fig.add_trace(go.Scatter(
        x=ms, y=theoretical, mode="lines",
        line=dict(color=_C_FP, width=1.8),
        name="1\u2212(1\u2212\u03b1)\u1d50", showlegend=True,
        hovertemplate="m=%{x}<br>FWER=%{y:.3f}<extra>Theoretical</extra>",
    ))

    fig.add_hline(y=alpha, line_dash="dot", line_color=t["muted"], line_width=0.8)
    fig.add_annotation(
        x=0.02, xref="paper", y=alpha, yref="y",
        text=f"\u03b1={alpha}", showarrow=False,
        font=dict(size=8, color=t["muted"]), yshift=10,
    )

    if empirical_fwer is not None and total > 0:
        fig.add_trace(go.Scatter(
            x=[m], y=[empirical_fwer], mode="markers",
            marker=dict(color=_C_BH, size=10, symbol="circle",
                        line=dict(color="#fff", width=1.5)),
            name=f"Empirical (n={total})", showlegend=True,
            hovertemplate=f"m={m}<br>FWER={empirical_fwer:.3f}<extra>Empirical</extra>",
        ))

    # Annotate key value
    theo_at_m = 1 - (1 - alpha) ** m
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text=f"At m={m}: theoretical FWER={theo_at_m:.1%}",
        showarrow=False,
        font=dict(size=9, color=t["annot_text"]),
        bgcolor=t["annot_bg"], bordercolor=t["annot_border"], borderwidth=1, borderpad=3,
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(size=8, color=t["annot_text"]),
            bgcolor="rgba(0,0,0,0)",
            x=0.02, y=0.60, xanchor="left", yanchor="top",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  p-value histogram (all accumulated p-values)
# ─────────────────────────────────────────────────────────────────────────────
def draw_mt_pvalue_hist(
    all_pvalues: list,
    alpha: float,
    m: int,
    file_drawer: bool = False,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    x_max = 0.055 if file_drawer else 1.0
    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(
            text="p-value", font=dict(size=10, color=t["label"])),
            range=[0, x_max],
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if len(all_pvalues) < 10:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    pvals = [p for p in all_pvalues if p < 0.05] if file_drawer else all_pvalues
    if len(pvals) == 0:
        return fig

    bin_size = 0.005 if file_drawer else 0.05
    fig.add_trace(go.Histogram(
        x=pvals,
        xbins=dict(start=0, end=x_max, size=bin_size),
        histnorm="probability density",
        marker=dict(
            color="rgba(56,189,248,0.35)",
            line=dict(color="#38bdf8", width=0.6),
        ),
        hovertemplate="p \u2248 %{x:.3f}<br>Density=%{y:.1f}<extra></extra>",
    ))

    if not file_drawer:
        # Correction threshold lines
        ranks = np.arange(1, max(m, 1) + 1)
        c_m = float(np.sum(1.0 / ranks)) if m > 0 else 1.0
        thresholds = {
            "None": (alpha, _C_FP),
            "Bonf": (alpha / max(m, 1), _C_BONF),
            "BH max": (alpha, _C_BH),           # max BH threshold = α
            "BY max": (alpha / c_m, _C_BY),      # max BY threshold = α/c_m
        }
        for label, (val, col) in thresholds.items():
            if val < 1:
                fig.add_vline(x=val, line_dash="dot", line_color=col, line_width=0.9)

    # Uniform expectation line under H₀
    if not file_drawer:
        fig.add_hline(y=1.0, line_dash="dash", line_color=t["muted"],
                      line_width=0.7)
        fig.add_annotation(
            x=0.75, y=1.0, yref="y",
            text="Uniform (all H\u2080)", showarrow=False,
            font=dict(size=8, color=t["muted"]), yshift=10,
        )

    return fig
