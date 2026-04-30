# =============================================================================
# Shared Plotly theme — layout constants, base figure builder, fig → UI helper
# =============================================================================

import plotly.graph_objects as go
from shiny import ui

# ── Colour tokens ─────────────────────────────────────────────────────────────
_TRANSPARENT = "rgba(0,0,0,0)"

# Dark mode
_GRID  = "#334155"
_AXIS  = "#94a3b8"
_LABEL = "#cbd5e1"

# Light mode
_LIGHT_GRID  = "#e2e8f0"
_LIGHT_AXIS  = "#64748b"
_LIGHT_LABEL = "#475569"

# ── Layout presets ────────────────────────────────────────────────────────────
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

# Plotly → HTML config (modebar hidden, responsive sizing)
_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


# ── Helper functions ──────────────────────────────────────────────────────────

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


def fig_to_ui(fig) -> ui.Tag:
    """Convert a Plotly Figure to a Shiny ui.HTML element."""
    html = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=_PLOTLY_CONFIG,
    )
    return ui.div(ui.HTML(html), class_="plotly-container")
