# =============================================================================
# Lord Of The Probability and Statistics - Interactive Shiny for Python App
# =============================================================================

from pathlib import Path
from shiny import App

from ui_layout import app_ui
from server import server

css_dir = Path(__file__).parent / "css"
app = App(app_ui, server, static_assets=css_dir)
