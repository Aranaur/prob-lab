# =============================================================================
# Bootstrap Explorer — Plotly chart helpers
# =============================================================================

import numpy as np
import plotly.graph_objects as go

from theme import _DARK_LAYOUT, _LIGHT_LAYOUT, _base_fig, _theme

# ── colour tokens ──────────────────────────────────────────────────────────────
_C_ORIG   = "#94a3b8"   # slate — original sample
_C_RESAMP = "#38bdf8"   # cyan  — resampled highlights
_C_THETA  = "#fbbf24"   # gold  — θ̂ original
_C_TRUE   = "#34d399"   # green — true θ
_C_HIST   = "#64748b"   # gray  — histogram fill

_CI_COLORS = {
    "Percentile":  "#38bdf8",
    "Normal":      "#f97316",
    "Basic":       "#a78bfa",
    "Studentized": "#f87171",
    "BCa":         "#34d399",
}

_REF_COLORS = {
    "t-test":   "#fbbf24",
    "Wilcoxon": "#fbbf24",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Original sample + resample encoding
# ═══════════════════════════════════════════════════════════════════════════════

def draw_boot_sample(sample, counts, theta_hat, true_theta, dark=True):
    """Scatter of original sample with resample-count size/colour encoding."""
    fig = _base_fig(dark)
    n = len(sample)
    xs = np.arange(1, n + 1)

    # Base layer (all points, faint)
    fig.add_trace(go.Scatter(
        x=xs, y=sample, mode="markers",
        marker=dict(color=_C_ORIG, size=6, opacity=0.3),
        name="Original", showlegend=True,
    ))

    # Resample overlay
    if counts is not None:
        mask = counts > 0
        if mask.any():
            sizes = 6 + counts[mask] * 4
            fig.add_trace(go.Scatter(
                x=xs[mask], y=sample[mask], mode="markers",
                marker=dict(
                    color=counts[mask],
                    colorscale=[[0, _C_ORIG], [1, _C_RESAMP]],
                    cmin=0, cmax=max(3, int(counts.max())),
                    size=sizes, opacity=0.8, line=dict(width=0),
                ),
                name="Resampled",
                text=[f"count={c}" for c in counts[mask]],
                hoverinfo="text+y",
            ))

    # θ̂ and true θ reference lines
    fig.add_hline(y=theta_hat,
                  line=dict(color=_C_THETA, width=1.5, dash="dash"),
                  annotation_text=f"\u03b8\u0302={theta_hat:.3f}",
                  annotation_position="top right")
    if true_theta is not None:
        fig.add_hline(y=true_theta,
                      line=dict(color=_C_TRUE, width=1, dash="dot"),
                      annotation_text=f"\u03b8={true_theta:.3f}",
                      annotation_position="bottom right")

    fig.update_layout(
        xaxis_title="Observation index", yaxis_title="Value",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bootstrap distribution histogram
# ═══════════════════════════════════════════════════════════════════════════════

def draw_boot_distribution(boot_stats, theta_hat, true_theta, cis,
                           bias, se_boot, bca_z0, bca_a, dark=True):
    """Histogram of θ̂* with CI boundary lines, bias / SE annotations."""
    fig = _base_fig(dark)
    ann_col = _theme(dark)["label"]

    if len(boot_stats) > 0:
        fig.add_trace(go.Histogram(
            x=boot_stats, nbinsx=50,
            marker_color=_C_HIST, opacity=0.6, name="\u03b8\u0302*",
        ))

    # θ̂ and true θ vertical lines
    fig.add_vline(x=theta_hat,
                  line=dict(color=_C_THETA, width=2, dash="dash"),
                  annotation_text="\u03b8\u0302", annotation_position="top left")
    if true_theta is not None:
        fig.add_vline(x=true_theta,
                      line=dict(color=_C_TRUE, width=1.5, dash="dot"),
                      annotation_text="\u03b8", annotation_position="top right")

    # CI shading + boundary lines
    for method, (lo, hi) in cis.items():
        col = _CI_COLORS.get(method, "#ffffff")
        fig.add_vrect(x0=lo, x1=hi, fillcolor=col, opacity=0.06,
                      line=dict(width=0))
        fig.add_vline(x=lo, line=dict(color=col, width=1, dash="dash"))
        fig.add_vline(x=hi, line=dict(color=col, width=1, dash="dash"))

    # Annotations
    ya = 0.98
    if bias is not None:
        bias_col = "#f87171" if abs(bias) > max(0.01 * se_boot, 1e-6) else ann_col
        fig.add_annotation(
            x=0.02, y=ya, xref="paper", yref="paper",
            text=f"Bias = {bias:.4f}",
            showarrow=False, font=dict(size=10, color=bias_col),
            xanchor="left", yanchor="top",
        )
        ya -= 0.06
    if se_boot is not None:
        fig.add_annotation(
            x=0.02, y=ya, xref="paper", yref="paper",
            text=f"SE<sub>boot</sub> = {se_boot:.4f}",
            showarrow=False, font=dict(size=10, color=ann_col),
            xanchor="left", yanchor="top",
        )
        ya -= 0.06
    if bca_z0 is not None and "BCa" in cis:
        fig.add_annotation(
            x=0.02, y=ya, xref="paper", yref="paper",
            text=f"BCa: z\u2080={bca_z0:.3f}, a={bca_a:.4f}",
            showarrow=False, font=dict(size=10, color=_CI_COLORS["BCa"]),
            xanchor="left", yanchor="top",
        )

    fig.update_layout(xaxis_title="\u03b8\u0302*", yaxis_title="Count",
                      showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CI comparison (forest plot)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_boot_ci_forest(cis, theta_hat, true_theta, dark=True):
    """Horizontal error bars for each CI method — forest-plot style."""
    fig = _base_fig(dark)

    methods = list(cis.keys())
    if not methods:
        return fig

    for method in methods:
        lo, hi = cis[method]
        col = _CI_COLORS.get(method, "#ffffff")
        # CI bar
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[method, method], mode="lines",
            line=dict(color=col, width=3),
            showlegend=False, hoverinfo="skip",
        ))
        # θ̂ diamond
        fig.add_trace(go.Scatter(
            x=[theta_hat], y=[method], mode="markers",
            marker=dict(color=col, size=8, symbol="diamond"),
            showlegend=False,
            text=[f"{method}: [{lo:.3f}, {hi:.3f}]  w={hi-lo:.3f}"],
            hoverinfo="text",
        ))
        # Width annotation
        fig.add_annotation(
            x=hi, y=method, text=f"  w={hi - lo:.3f}",
            showarrow=False, font=dict(size=9, color=col), xanchor="left",
        )

    # True θ reference
    if true_theta is not None:
        fig.add_vline(x=true_theta,
                      line=dict(color=_C_TRUE, width=1.5, dash="dash"),
                      annotation_text="true \u03b8",
                      annotation_position="top right")

    fig.add_vline(x=theta_hat,
                  line=dict(color=_C_THETA, width=1, dash="dot"))

    fig.update_layout(xaxis_title="\u03b8", yaxis=dict(type="category"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Coverage simulation
# ═══════════════════════════════════════════════════════════════════════════════

def draw_boot_coverage(coverage_dict, total, conf_level, dark=True):
    """Bar chart: empirical coverage % per CI method with nominal reference."""
    fig = _base_fig(dark)
    ann_col = _theme(dark)["label"]

    if not coverage_dict or total == 0:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Run experiments to see coverage",
            showarrow=False, font=dict(size=13, color=ann_col),
        )
        fig.update_layout(xaxis_title="CI Method", yaxis_title="Coverage %")
        return fig

    methods   = list(coverage_dict.keys())
    coverages = [coverage_dict[m] / total * 100 for m in methods]
    colors    = [_CI_COLORS.get(m, _C_HIST) for m in methods]

    fig.add_trace(go.Bar(
        x=methods, y=coverages, marker_color=colors, width=0.5,
        text=[f"{c:.1f}%" for c in coverages], textposition="outside",
    ))

    # Nominal reference + ±2 SE band
    fig.add_hline(y=conf_level,
                  line=dict(color=_C_THETA, width=1.5, dash="dash"),
                  annotation_text=f"Nominal {conf_level:.0f}%",
                  annotation_position="top right")
    p  = conf_level / 100
    se = np.sqrt(p * (1 - p) / total) * 100
    fig.add_hrect(y0=conf_level - 2 * se, y1=conf_level + 2 * se,
                  fillcolor=_C_THETA, opacity=0.08, line=dict(width=0))

    fig.update_layout(
        xaxis_title="CI Method", yaxis_title="Coverage %",
        yaxis=dict(range=[max(0, min(coverages) - 10), 105]),
        showlegend=False,
    )
    fig.add_annotation(
        x=0.98, y=0.02, xref="paper", yref="paper",
        text=f"n = {total} experiments",
        showarrow=False, font=dict(size=10, color=ann_col),
        xanchor="right", yanchor="bottom",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Convergence analysis — Non-coverage rate vs N
# ═══════════════════════════════════════════════════════════════════════════════

def draw_boot_convergence(results, alpha, dark=True):
    """Line chart: non-coverage rate vs sample-size N for each CI method.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys: "N", "method", "fpr" (non-coverage rate), "se".
    alpha : float
        Nominal significance level (e.g. 0.05).
    """
    fig = _base_fig(dark)
    ann_col = _theme(dark)["label"]

    if not results:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Click \u201cRun Convergence\u201d to start the simulation",
            showarrow=False, font=dict(size=13, color=ann_col),
        )
        fig.update_layout(xaxis_title="Sample size N",
                          yaxis_title="Non-coverage rate (1 \u2212 coverage)")
        return fig

    # Group by method
    from collections import defaultdict
    grouped = defaultdict(lambda: {"N": [], "fpr": [], "se": []})
    for r in results:
        g = grouped[r["method"]]
        g["N"].append(r["N"])
        g["fpr"].append(r["fpr"])
        g["se"].append(r["se"])

    max_y = max(alpha * 3, 0.15)
    for method, g in grouped.items():
        ns   = g["N"]
        fprs = np.array(g["fpr"])
        ses  = np.array(g["se"])

        col = _CI_COLORS.get(method, _REF_COLORS.get(method, "#ffffff"))
        is_ref = method in _REF_COLORS

        # Confidence band (±1.96 SE)
        upper = fprs + 1.96 * ses
        lower = np.maximum(fprs - 1.96 * ses, 0.0)
        max_y = max(max_y, np.max(upper))

        fig.add_trace(go.Scatter(
            x=ns, y=upper.tolist(), mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=ns, y=lower.tolist(), mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=_rgba(col, 0.12),
            showlegend=False, hoverinfo="skip",
        ))

        # Main line
        fig.add_trace(go.Scatter(
            x=ns, y=fprs.tolist(), mode="lines+markers",
            line=dict(color=col, width=2.5,
                      dash="dot" if is_ref else "solid"),
            marker=dict(size=6 if not is_ref else 5),
            name=method,
            text=[f"{method}: {f:.3f} \u00b1 {s:.3f}<br>N={n}"
                  for n, f, s in zip(ns, fprs, ses)],
            hoverinfo="text",
        ))

    # Nominal α reference line
    fig.add_hline(
        y=alpha,
        line=dict(color="#ef4444", width=1.5, dash="dash"),
        annotation_text=f"\u03b1 = {alpha}",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=11),
    )

    fig.update_layout(
        xaxis_title="Sample size N",
        yaxis_title="Non-coverage rate (1 \u2212 coverage)",
        xaxis=dict(type="log", dtick=1),
        yaxis=dict(range=[0, max_y * 1.05]),
        showlegend=True,
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0)"),
    )

    return fig


def _rgba(hex_color: str, opacity: float) -> str:
    """Convert '#RRGGBB' to 'rgba(R,G,B,opacity)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"
