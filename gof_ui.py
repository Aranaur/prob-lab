# =============================================================================
# GoF Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def gof_panel() -> ui.Tag:
    return ui.nav_panel(
        "GoF Explorer",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconception: "),
                    "A non-significant test means the data follow the hypothesised distribution.",
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "The test may simply lack ",
                    ui.tags.em("power"),
                    " to detect the deviation \u2014 especially with small n.",
                    class_="info-banner-text",
                ),

                # Test selector
                ui.input_select(
                    "gof_test",
                    ui.TagList(
                        "Test",
                        tip("Choose a goodness-of-fit test. Each has different "
                            "assumptions and power characteristics."),
                    ),
                    choices={
                        "ks1":  "Kolmogorov (one-sample)",
                        "ks2":  "Kolmogorov\u2013Smirnov (two-sample)",
                        "chi2": "\u03c7\u00b2 Goodness of Fit",
                        "sw":   "Shapiro\u2013Wilk (normality)",
                    },
                    selected="ks1", width="100%",
                ),

                # Significance level
                ui.input_slider(
                    "gof_alpha",
                    ui.TagList("\u03b1 (significance level)",
                               tip("Probability of rejecting H\u2080 when it is true.")),
                    min=0.01, max=0.20, value=0.05, step=0.01, width="100%",
                ),

                # Dynamic parameter block
                ui.output_ui("gof_params_block"),

                # ── Sampling controls ────────────────────────────────────────
                ui.div(
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("gof_btn_sample_1",   "\u00d71",   class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("gof_btn_sample_50",  "\u00d750",  class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("gof_btn_sample_100", "\u00d7100", class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("gof_speed_minus", "\u2212", class_="btn-ctrl btn-pm"),
                            ui.input_action_button("gof_btn_play", "Play", class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("gof_speed_plus", "+", class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.input_action_button("gof_btn_reset", "Reset", class_="btn-ctrl btn-reset btn-full"),
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

            # ── RIGHT MAIN PANEL ─────────────────────────────────────────────
            ui.div(

                # Stats row
                ui.div(
                    ui.div(
                        ui.div(
                            ui.output_text("gof_stat_label", inline=True), " ",
                            tip("The computed test statistic for the current sample."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("gof_stat_value", inline=True), class_="stat-value pw-d"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "p-VALUE ",
                            tip("Probability of observing a statistic as extreme "
                                "as the current one under H\u2080."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("gof_pvalue", inline=True), class_="stat-value coverage"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "REJECT RATE\u00a0",
                            tip("Fraction of tests where H\u2080 was rejected at \u03b1."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("gof_reject_rate", inline=True), class_="stat-value included"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "TOTAL EXPERIMENTS\u00a0",
                            tip("Total number of goodness-of-fit tests performed."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("gof_total_tests", inline=True), class_="stat-value total"),
                        class_="stat-card",
                    ),
                    class_="stats-row",
                ),

                # Charts
                ui.div(
                    # Left column: null distribution + p-value histogram
                    ui.div(
                        ui.div(
                            ui.div("NULL DISTRIBUTION & TEST STATISTIC", class_="card-title"),
                            ui.output_ui("gof_null_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("p-VALUE DISTRIBUTION", class_="card-title"),
                            ui.output_ui("gof_pval_hist"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: main chart (ECDF / bars / Q-Q)
                    ui.div(
                        ui.div(
                            ui.div(ui.output_text("gof_main_chart_title", inline=True),
                                   class_="card-title"),
                            ui.output_ui("gof_main_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-right",
                    ),
                    class_="charts-area",
                ),

                class_="main-panel",
            ),

            class_="app-body",
        ),
    )
