# =============================================================================
# UI layout definition (app_ui)
# =============================================================================

from shiny import ui
from utils import tip
from pvalue_ui import pvalue_panel
from power_ui import power_panel
from gof_ui import gof_panel
from np_ui import np_panel
from mt_ui import mt_panel
from seq_ui import seq_panel
from vr_ui import vr_panel
from boot_ui import boot_panel
from bayes_ui import bayes_panel
from ci_ui import ci_panel

app_ui = ui.page_fluid(

    # ── Shared head resources ─────────────────────────────────────────────────
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.HTML('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'),
        ui.HTML('<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'),
    ),

    # JS helpers — play/pause button class + dark/light theme toggle
    ui.tags.head(ui.tags.script("""
        Shiny.addCustomMessageHandler('togglePlayClass', function(msg) {
            var btn = document.getElementById('ci_btn_play');
            if (msg.playing) {
                btn.classList.remove('btn-play');
                btn.classList.add('btn-pause');
            } else {
                btn.classList.remove('btn-pause');
                btn.classList.add('btn-play');
            }
        });

        /* Theme toggle: JS handles body class + button label instantly;
           the Shiny server reacts to the click count for plot re-rendering. */
        $(document).on('click', '#theme_toggle', function() {
            document.body.classList.toggle('light-mode');
            var isDark = !document.body.classList.contains('light-mode');
            this.textContent = isDark ? '\u2600 Light' : '\u263d Dark';
        });
    """)),

    # ── Navigation ────────────────────────────────────────────────────────────
    ui.navset_pill(

        # ── Tab 1: CI Explorer ────────────────────────────────────────────────
        ci_panel(),

        # ── Tab 2: p-value Explorer ────────────────────────────────────────────
        pvalue_panel(),

        # ── Tab 3: Power Explorer ─────────────────────────────────────────────
        power_panel(),

        # ── Tab 4: GoF Explorer ──────────────────────────────────────────────
        gof_panel(),

        # ── Tab 5: Nonparametric Explorer ────────────────────────────────────
        np_panel(),

        # ── Tab 6: Multiple Testing Explorer ────────────────────────────────
        mt_panel(),

        # ── Tab 7: Sequential Testing Explorer ──────────────────────────────
        seq_panel(),

        # ── Tab 8: Variance Reduction Explorer ──────────────────────────────
        vr_panel(),

        # ── Tab 9: Bootstrap Explorer ──────────────────────────────────────
        boot_panel(),

        # ── Tab 10: Bayesian Explorer ──────────────────────────────────────
        bayes_panel(),

        ui.nav_spacer(),
        ui.nav_control(
            ui.input_action_button(
                "theme_toggle", "\u2600 Light",
                class_="btn-ctrl btn-theme",
            )
        ),

        id="main_nav",
    ),
)
