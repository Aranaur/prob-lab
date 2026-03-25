# =============================================================================
# Confidence Intervals Explorer - Interactive Shiny for Python App
# Inspired by https://rpsychologist.com/d3/ci/
# =============================================================================

from pathlib import Path
from shiny import App

from ui_layout import app_ui
from server import server

css_dir = Path(__file__).parent / "css"
app = App(app_ui, server, static_assets=css_dir)
