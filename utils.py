# =============================================================================
# Utility helpers: tip() tooltip and shared constants
# =============================================================================

from shiny import ui


# ── Tooltip helper ───────────────────────────────────────────────────────────
_INFO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" '
    'fill="currentColor" viewBox="0 0 16 16">'
    '<path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16"/>'
    '<path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 '
    "3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2"
    '.176-.492.246-.686.246-.275 0-.375-.193-.304-.533zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0"/>'
    "</svg>"
)


def tip(text: str) -> ui.Tag:
    """Return an info-icon with a native hover tooltip."""
    return ui.HTML(f'<span class="tip-icon" title="{text}">{_INFO_SVG}</span>')
