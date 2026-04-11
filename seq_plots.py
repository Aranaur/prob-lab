# =============================================================================
# Peeking & Sequential Testing Explorer — Plotly chart builders
# =============================================================================

import numpy as np
from scipy import stats
import plotly.graph_objects as go

from plots import _DARK_LAYOUT, _LIGHT_LAYOUT, _base_fig, _theme

# ── Color tokens ─────────────────────────────────────────────────────────────
_C_PEEK   = "#f87171"   # red – peeking / Type I inflation
_C_SEQ    = "#38bdf8"   # cyan – sequential (controlled)
_C_TRAJ   = "#94a3b8"   # gray – background trajectories
_C_LAST   = "#fbbf24"   # yellow – highlighted (last) trajectory
_C_OBF    = "#38bdf8"   # cyan
_C_POC    = "#f97316"   # orange
_C_HP     = "#c084fc"   # purple
_C_ALPHA  = "#f87171"   # red – nominal alpha line
_C_STOP   = "#34d399"   # green – success / stopped
_C_FUTL   = "#64748b"   # muted gray – futility zone


# ─────────────────────────────────────────────────────────────────────────────
# 1. p-value trajectory — main chart (right column, tall)
# ─────────────────────────────────────────────────────────────────────────────
def draw_seq_trajectory(
    trajectories: list[np.ndarray] | None,
    n_min: int,
    N: int,
    alpha: float,
    boundary_pvals: np.ndarray | None = None,
    look_ns: np.ndarray | None = None,
    stop_ns: list[int | None] | None = None,
    dark: bool = True,
) -> go.Figure:
    """
    trajectories: list of arrays, each shape (n_points,) p-values at n_min..N or at look_ns
    boundary_pvals: sequential boundary expressed as p-value equivalents at each look
    look_ns: sample sizes for each look (sequential mode) or arange(n_min, N+1) (peeking)
    stop_ns: list of sample sizes where each experiment stopped (None = didn't stop)
    """
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(
            text="Sample size (n)",
            font=dict(size=10, color=t["label"]),
        )),
        yaxis=dict(
            showticklabels=True, showgrid=True,
            gridcolor=t["grid"], gridwidth=0.3,
            tickfont=dict(size=9, color=t["axis"]),
            title=dict(text="p-value", font=dict(size=10, color=t["label"])),
            range=[0, 1.05], zeroline=False,
        ),
    )

    if trajectories is None or len(trajectories) == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Press Sample to begin", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    xs = look_ns if look_ns is not None else np.arange(n_min, N + 1)

    # "Continue" zone shading between alpha and boundary (if sequential)
    if boundary_pvals is not None and len(boundary_pvals) == len(xs):
        # Success zone (below boundary)
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([boundary_pvals, np.zeros(len(xs))]),
            fill="toself", fillcolor="rgba(52,211,153,0.06)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            showlegend=False,
        ))

    # Background trajectories (all but last)
    for i, traj in enumerate(trajectories[:-1]):
        n_pts = min(len(traj), len(xs))
        fig.add_trace(go.Scatter(
            x=xs[:n_pts], y=traj[:n_pts], mode="lines",
            line=dict(color=_C_TRAJ, width=0.6), opacity=0.3,
            hoverinfo="skip", showlegend=False,
        ))

    # Last trajectory (highlighted)
    last = trajectories[-1]
    n_pts = min(len(last), len(xs))
    fig.add_trace(go.Scatter(
        x=xs[:n_pts], y=last[:n_pts], mode="lines",
        line=dict(color=_C_LAST, width=2),
        name="Current experiment", showlegend=True,
        hovertemplate="n=%{x}<br>p=%{y:.4f}<extra></extra>",
    ))

    # α line
    fig.add_hline(y=alpha, line_dash="dash", line_color=_C_ALPHA, line_width=1.2)
    fig.add_annotation(
        x=1, xref="paper", y=alpha, yref="y",
        text=f"\u03b1={alpha}", showarrow=False,
        font=dict(size=9, color=_C_ALPHA), xanchor="right", yshift=10,
    )

    # Sequential boundary line (as p-values)
    if boundary_pvals is not None and len(boundary_pvals) == len(xs):
        fig.add_trace(go.Scatter(
            x=xs, y=boundary_pvals, mode="lines+markers",
            line=dict(color=_C_SEQ, width=1.5, dash="dot"),
            marker=dict(size=4, color=_C_SEQ),
            name="Sequential boundary", showlegend=True,
            hovertemplate="n=%{x}<br>p\u209b=%{y:.4f}<extra>Boundary</extra>",
        ))

    # Stop markers
    has_stop = False
    is_seq = boundary_pvals is not None and len(boundary_pvals) == len(xs)
    bg_col = _C_SEQ if is_seq else _C_PEEK

    if stop_ns is not None:
        for i, sn in enumerate(stop_ns):
            if sn is not None and i < len(trajectories):
                traj = trajectories[i]
                idx = None
                for j, x in enumerate(xs):
                    if x >= sn:
                        idx = j
                        break
                if idx is not None and idx < len(traj):
                    has_stop = True
                    col = _C_LAST if i == len(trajectories) - 1 else bg_col
                    fig.add_trace(go.Scatter(
                        x=[xs[idx]], y=[traj[idx]], mode="markers",
                        marker=dict(size=8, color=col, symbol="diamond",
                                    line=dict(width=1, color="#fff")),
                        showlegend=False, hoverinfo="skip",
                    ))

    if has_stop:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=bg_col, symbol="diamond",
                        line=dict(width=1, color="#fff")),
            name="Stopped early", showlegend=True,
        ))

    # Info fraction top axis annotation
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"Information fraction: n/N (N={N})",
        showarrow=False, font=dict(size=10, color=t["muted"]),
        xanchor="left", yanchor="top",
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(size=8, color=t["annot_text"]),
            bgcolor="rgba(0,0,0,0)",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Boundary shape — z-statistic boundaries for all methods
# ─────────────────────────────────────────────────────────────────────────────
def draw_seq_boundary(
    K: int,
    alpha: float,
    obf_z: np.ndarray,
    pocock_z: float,
    hp_z: np.ndarray,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(
            text="Look number",
            font=dict(size=10, color=t["label"]),
        ), dtick=1),
        yaxis=dict(
            showticklabels=True, showgrid=True,
            gridcolor=t["grid"], gridwidth=0.3,
            tickfont=dict(size=9, color=t["axis"]),
            title=dict(text="z-boundary", font=dict(size=10, color=t["label"])),
            zeroline=False,
        ),
    )

    looks = np.arange(1, K + 1)

    fig.add_trace(go.Scatter(
        x=looks, y=obf_z, mode="lines+markers",
        line=dict(color=_C_OBF, width=2),
        marker=dict(size=5, color=_C_OBF),
        name="O\u2019Brien-Fleming", showlegend=True,
        hovertemplate="Look %{x}<br>z=%{y:.3f}<br>p=%{customdata:.4f}<extra>OBF</extra>",
        customdata=2 * (1 - stats.norm.cdf(obf_z)),
    ))

    fig.add_trace(go.Scatter(
        x=looks, y=np.full(K, pocock_z), mode="lines+markers",
        line=dict(color=_C_POC, width=2, dash="dash"),
        marker=dict(size=5, color=_C_POC),
        name="Pocock", showlegend=True,
        hovertemplate="Look %{x}<br>z=%{y:.3f}<br>p=%{customdata:.4f}<extra>Pocock</extra>",
        customdata=2 * (1 - stats.norm.cdf(np.full(K, pocock_z))),
    ))

    fig.add_trace(go.Scatter(
        x=looks, y=hp_z, mode="lines+markers",
        line=dict(color=_C_HP, width=2, dash="dot"),
        marker=dict(size=5, color=_C_HP),
        name="Haybittle-Peto", showlegend=True,
        hovertemplate="Look %{x}<br>z=%{y:.3f}<br>p=%{customdata:.4f}<extra>H-P</extra>",
        customdata=2 * (1 - stats.norm.cdf(hp_z)),
    ))

    # z_alpha/2 reference
    z_a2 = stats.norm.ppf(1 - alpha / 2)
    fig.add_hline(y=z_a2, line_dash="dot", line_color=_C_ALPHA, line_width=0.8)
    fig.add_annotation(
        x=1, xref="paper", y=z_a2, yref="y",
        text=f"z_\u03b1/2={z_a2:.2f}", showarrow=False,
        font=dict(size=8, color=_C_ALPHA), xanchor="right", yshift=-10,
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(size=8, color=t["annot_text"]),
            bgcolor="rgba(0,0,0,0)",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stopping distribution — histogram of when experiments stopped
# ─────────────────────────────────────────────────────────────────────────────
def draw_seq_stopping_hist(
    stop_ns_peek: list[int],
    stop_ns_seq: list[int],
    N: int,
    dark: bool = True,
) -> go.Figure:
    t = _theme(dark)
    _ax = _DARK_LAYOUT["xaxis"] if dark else _LIGHT_LAYOUT["xaxis"]

    fig = _base_fig(
        dark=dark,
        xaxis=dict(**_ax, title=dict(
            text="Sample size at stop",
            font=dict(size=10, color=t["label"]),
        )),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    if len(stop_ns_peek) < 3 and len(stop_ns_seq) < 3:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    if len(stop_ns_peek) >= 3:
        fig.add_trace(go.Histogram(
            x=stop_ns_peek,
            histnorm="probability density",
            marker=dict(color="rgba(248,113,113,0.4)",
                        line=dict(color=_C_PEEK, width=0.6)),
            name="Peeking", showlegend=True,
        ))

    if len(stop_ns_seq) >= 3:
        fig.add_trace(go.Histogram(
            x=stop_ns_seq,
            histnorm="probability density",
            marker=dict(color="rgba(56,189,248,0.4)",
                        line=dict(color=_C_SEQ, width=0.6)),
            name="Sequential", showlegend=True,
        ))

    if len(stop_ns_peek) >= 3 and len(stop_ns_seq) >= 3:
        fig.update_layout(barmode="overlay")

    # N_max reference line
    fig.add_vline(x=N, line_dash="dash", line_color=t["muted"], line_width=0.8)
    fig.add_annotation(
        x=N, y=1, yref="paper",
        text=f"N={N}", showarrow=False,
        font=dict(size=8, color=t["muted"]), xshift=5, yanchor="top",
    )

    # ASN annotations
    if len(stop_ns_peek) >= 3:
        mean_peek = np.mean(stop_ns_peek)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            text=f"<span style='color:{_C_PEEK}'>Peek E[n]={mean_peek:.0f}</span>",
            showarrow=False, font=dict(size=9, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"],
            borderwidth=1, borderpad=3,
        )
    if len(stop_ns_seq) >= 3:
        mean_seq = np.mean(stop_ns_seq)
        y_pos = 0.88 if len(stop_ns_peek) >= 3 else 0.98
        fig.add_annotation(
            xref="paper", yref="paper", x=0.98, y=y_pos,
            xanchor="right", yanchor="top",
            text=f"<span style='color:{_C_SEQ}'>Seq E[n]={mean_seq:.0f}</span>",
            showarrow=False, font=dict(size=9, color=t["annot_text"]),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"],
            borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(size=8, color=t["annot_text"]),
            bgcolor="rgba(0,0,0,0)",
            x=0.02, y=0.98, xanchor="left", yanchor="top",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Type I error comparison — bar chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_seq_error_bars(
    alpha: float,
    peek_fwer: float | None,
    seq_fwer: float | None,
    fixed_power: float | None = None,
    seq_power: float | None = None,
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
            title=dict(text="Rate", font=dict(size=10, color=t["label"])),
            range=[0, max(0.15, (peek_fwer or 0) * 1.3, 0.06)],
            zeroline=False,
        ),
    )

    if peek_fwer is None and seq_fwer is None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="Collecting data\u2026", showarrow=False,
            font=dict(size=13, color=t["muted"]),
        )
        return fig

    labels = []
    values = []
    colors = []

    if peek_fwer is not None:
        labels.append("Peeking")
        values.append(peek_fwer)
        colors.append(_C_PEEK)

    if seq_fwer is not None:
        labels.append("Sequential")
        values.append(seq_fwer)
        colors.append(_C_SEQ)

    if fixed_power is not None:
        labels.append("Fixed\nPower")
        values.append(fixed_power)
        colors.append("#94a3b8")

    if seq_power is not None:
        labels.append("Seq\nPower")
        values.append(seq_power)
        colors.append(_C_OBF)

    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors, opacity=0.85,
        showlegend=False,
        hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
    ))

    # α target line
    fig.add_hline(y=alpha, line_dash="dash", line_color=_C_ALPHA, line_width=1.2)
    fig.add_annotation(
        x=1, xref="paper", y=alpha, yref="y",
        text=f"\u03b1={alpha}", showarrow=False,
        font=dict(size=8, color=_C_ALPHA), xanchor="right", yshift=8,
    )

    # Inflation annotation
    if peek_fwer is not None and alpha > 0:
        ratio = peek_fwer / alpha
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            text=f"Peeking inflated \u03b1 by {ratio:.1f}\u00d7",
            showarrow=False, font=dict(size=9, color=_C_PEEK),
            bgcolor=t["annot_bg"], bordercolor=t["annot_border"],
            borderwidth=1, borderpad=3,
        )

    return fig
