# =============================================================================
# Server: reactive logic, event handlers, render functions
# =============================================================================

from shiny import reactive, render, ui

from pvalue_server import pvalue_server
from power_server import power_server
from gof_server import gof_server
from np_server import np_server
from mt_server import mt_server
from seq_server import seq_server
from vr_server import vr_server
from boot_server import boot_server
from bayes_server import bayes_server

from ci_server import ci_server



def server(input, output, session):

    # ── Theme state ────────────────────────────────────────────────────────
    is_dark = reactive.value(True)

    @reactive.effect
    @reactive.event(input.theme_toggle)
    def _toggle_theme():
        # JS handles body class + button label; server just tracks state for plots
        is_dark.set(input.theme_toggle() % 2 == 0)

    pvalue_server(input, output, session, is_dark)
    power_server(input, output, session, is_dark)
    gof_server(input, output, session, is_dark)
    np_server(input, output, session, is_dark)
    mt_server(input, output, session, is_dark)
    seq_server(input, output, session, is_dark)
    vr_server(input, output, session, is_dark)
    boot_server(input, output, session, is_dark)
    bayes_server(input, output, session, is_dark)

    ci_server(input, output, session, is_dark)
