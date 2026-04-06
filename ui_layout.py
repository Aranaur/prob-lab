# =============================================================================
# UI layout definition (app_ui)
# =============================================================================

from shiny import ui
from utils import tip
from pvalue_ui import pvalue_panel
from power_ui import power_panel

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
            var btn = document.getElementById('btn_play');
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
        ui.nav_panel(
            "CI Explorer",

            ui.div(

                # ── LEFT SIDEBAR: controls ────────────────────────────────────
                ui.div(

                    # Misconception banner
                    ui.div(
                        ui.tags.i(class_="info-icon"),
                        ui.tags.strong(" Common Misconception: "),
                        "A ", ui.output_text("conf_pct", inline=True),
                        " CI does NOT mean a ",
                        ui.output_text("conf_pct2", inline=True),
                        " probability the true parameter lies within it.",
                        ui.tags.br(),
                        ui.tags.strong("Reality: "),
                        "If we repeat sampling many times, ",
                        ui.output_text("conf_pct3", inline=True),
                        " of intervals will contain the true parameter.",
                        class_="info-banner-text",
                    ),

                    # Confidence level slider
                    ui.input_slider(
                        "conf_level",
                        ui.TagList("Confidence Level (%)", tip("The probability that the interval estimation procedure will produce an interval containing the true parameter.")),
                        min=50, max=99, value=95, step=1, width="100%",
                    ),

                    # Distribution selector
                    ui.input_select(
                        "pop_dist",
                        ui.TagList("Population Distribution", tip("The theoretical probability distribution from which random samples are drawn.")),
                        choices={
                            "normal":      "Normal",
                            "uniform":     "Uniform",
                            "exponential": "Exponential",
                            "lognormal":   "Log-normal  (skewed right)",
                            "poisson":     "Poisson  (counts)",
                            "binomial":    "Binomial  (counts)",
                        },
                        selected="normal", width="100%",
                    ),

                    # Dynamic distribution parameters
                    ui.output_ui("dynamic_params"),

                    # Statistic selector (dynamic — adds Proportion for Binomial)
                    ui.output_ui("ci_statistic_ui"),

                    # Percentile level — shown only when Percentile is selected
                    ui.output_ui("ci_percentile_param"),

                    # CI method selector
                    ui.input_select(
                        "ci_method",
                        ui.TagList("CI Method", tip(
                            "t-interval: uses sample s, assumes normal sampling distribution. "
                            "z-interval: uses true population \u03c3 (known). "
                            "Bootstrap: resamples from the observed data \u2014 no distributional assumptions."
                        )),
                        choices={
                            "t":         "t-interval  (unknown \u03c3)",
                            "z":         "z-interval  (known \u03c3)",
                            "bootstrap": "Bootstrap   (percentile, B\u200a=\u200a500)",
                        },
                        selected="t",
                        width="100%",
                    ),



                    # Sampling controls
                    ui.div(
                        # Row 1: label + 3 equal sample buttons
                        ui.div(
                            ui.tags.span("Sample:", class_="btn-row-label"),
                            ui.input_action_button("btn_sample_1",   "\u00d71",   class_="btn-ctrl btn-sample btn-flex"),
                            ui.input_action_button("btn_sample_50",  "\u00d750",  class_="btn-ctrl btn-sample btn-flex"),
                            ui.input_action_button("btn_sample_100", "\u00d7100", class_="btn-ctrl btn-sample btn-flex"),
                            class_="sidebar-btn-row",
                        ),
                        # Row 3: speed + play
                        ui.div(
                            ui.div(
                                ui.tags.label("Speed"),
                                ui.input_action_button("speed_minus", "\u2212", class_="btn-ctrl btn-pm"),
                                ui.input_action_button("btn_play", "Play", class_="btn-ctrl btn-play btn-flex"),
                                ui.input_action_button("speed_plus", "+", class_="btn-ctrl btn-pm"),
                                class_="ctrl-group ctrl-group-full",
                            ),
                            class_="sidebar-btn-row",
                        ),
                        # Row 4: reset
                        ui.div(
                            ui.input_action_button("btn_reset", "Reset", class_="btn-ctrl btn-reset btn-full"),
                            class_="sidebar-btn-row",
                        ),
                        class_="sidebar-controls",
                    ),

                    # Footer
                    ui.div(
                        ui.tags.a("LinkedIn", href="https://www.linkedin.com/in/ihormiroshnychenko/", target="_blank"),
                        " \u2022 ",
                        ui.tags.a("Telegram", href="https://t.me/araprof", target="_blank"),
                        " \u2022 ",
                        ui.tags.a("Website", href="https://aranaur.rbind.io/", target="_blank"),
                        class_="footer-links",
                    ),

                    class_="sidebar",
                ),

                # ── RIGHT MAIN PANEL: stats + charts ──────────────────────────
                ui.div(

                    # Stats row
                    ui.div(
                        ui.div(
                            ui.div("CI COVERAGE ", tip("Percentage of all generated CIs that contain the true \u03bc."), class_="stat-label"),
                            ui.div(ui.output_text("cov_rate", inline=True), class_="stat-value coverage"),
                            class_="stat-card",
                        ),
                        ui.div(
                            ui.div(
                                ui.output_text("stat_label_inc", inline=True), " ",
                                tip("Count of intervals where the true parameter falls inside the CI."),
                                class_="stat-label",
                            ),
                            ui.div(ui.output_text("num_covered", inline=True), class_="stat-value included"),
                            class_="stat-card",
                        ),
                        ui.div(
                            ui.div(
                                ui.output_text("stat_label_miss", inline=True), " ",
                                tip("Count of intervals where the true parameter falls outside the CI."),
                                class_="stat-label",
                            ),
                            ui.div(ui.output_text("num_missed", inline=True), class_="stat-value missed"),
                            class_="stat-card",
                        ),
                        ui.div(
                            ui.div("SAMPLES DRAWN ", tip("Total number of random samples generated so far."), class_="stat-label"),
                            ui.div(ui.output_text("num_total", inline=True), class_="stat-value total"),
                            class_="stat-card",
                        ),
                        class_="stats-row",
                    ),

                    # Charts area
                    ui.div(
                        # Left column: 3 small charts (CLT on top)
                        ui.div(
                            ui.div(
                                ui.div(ui.output_text("stat_plot_title", inline=True), class_="card-title"),
                                ui.output_ui("means_plot"),
                                class_="glass-card chart-card",
                            ),
                            ui.div(
                                ui.div(ui.output_text("prop_plot_title", inline=True), class_="card-title"),
                                ui.output_ui("prop_plot"),
                                class_="glass-card chart-card",
                            ),
                            ui.div(
                                ui.div("CI WIDTH DISTRIBUTION", class_="card-title"),
                                ui.output_ui("width_plot"),
                                class_="glass-card chart-card",
                            ),
                            class_="charts-col-left",
                        ),
                        # Right column: population distribution + CI chart
                        ui.div(
                            ui.div(
                                ui.div("POPULATION DISTRIBUTION", class_="card-title"),
                                ui.output_ui("population_plot"),
                                class_="glass-card chart-card chart-card-pop",
                            ),
                            ui.div(
                                ui.div("CONFIDENCE INTERVALS", class_="card-title"),
                                ui.output_ui("ci_plot"),
                                class_="glass-card chart-card chart-card-ci",
                            ),
                            class_="charts-col-right",
                        ),
                        class_="charts-area",
                    ),

                    class_="main-panel",
                ),

                class_="app-body",
            ),
        ),  # end nav_panel CI Explorer

        # ── Tab 2: p-value Explorer ────────────────────────────────────────────
        pvalue_panel(),

        # ── Tab 3: Power Explorer ─────────────────────────────────────────────
        power_panel(),

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
