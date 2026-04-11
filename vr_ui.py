# =============================================================================
# Variance Reduction Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def vr_panel() -> ui.Tag:
    return ui.nav_panel(
        "Variance Reduction",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner (dynamic)
                ui.output_ui("vr_misconception_banner"),


                # Method selector
                ui.input_select(
                    "vr_mode",
                    ui.TagList(
                        "Method",
                        tip(
                            "CUPED: reduces variance via a pre-experiment covariate "
                            "correlated with the outcome. "
                            "VWE: down-weights high-variance \u2018power users\u2019 "
                            "for a more stable treatment-effect estimate."
                        ),
                    ),
                    choices={
                        "cuped": "CUPED  (covariate adjustment)",
                        "vwe":   "VWE   (variance weighting)",
                    },
                    selected="cuped", width="100%",
                ),

                # Dynamic mode-specific controls
                ui.output_ui("vr_mode_params"),

                # Common parameters
                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "vr_n",
                            ui.TagList(
                                "n (per group)",
                                tip("Sample size in each arm of the experiment."),
                            ),
                            value=200, min=20, max=5000, step=10, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric(
                            "vr_delta",
                            ui.TagList(
                                "\u03b4 (true effect)",
                                tip("Set to 0 for Type\u00a0I error rate; "
                                    "set >0 to observe power."),
                            ),
                            value=0.3, step=0.05, width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),

                ui.input_slider(
                    "vr_alpha",
                    ui.TagList(
                        "\u03b1",
                        tip("Significance level for two-sided tests."),
                    ),
                    min=0.01, max=0.20, value=0.05, step=0.01, width="100%",
                ),

                # VWE presets (rendered only in VWE mode)
                ui.output_ui("vr_presets"),

                # ── Sampling controls ────────────────────────────────────────
                ui.div(
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("vr_btn_sample_1", "\u00d71",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("vr_btn_sample_50", "\u00d750",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("vr_btn_sample_100", "\u00d7100",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("vr_speed_minus", "\u2212",
                                                   class_="btn-ctrl btn-pm"),
                            ui.input_action_button("vr_btn_play", "Play",
                                                   class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("vr_speed_plus", "+",
                                                   class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.input_action_button("vr_btn_reset", "Reset",
                                               class_="btn-ctrl btn-reset btn-full"),
                        class_="sidebar-btn-row",
                    ),
                    class_="sidebar-controls",
                ),

                # Footer
                ui.div(
                    ui.tags.a("LinkedIn",
                              href="https://www.linkedin.com/in/ihormiroshnychenko/",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Telegram", href="https://t.me/araprof",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Website", href="https://aranaur.rbind.io/",
                              target="_blank"),
                    class_="footer-links",
                ),

                class_="sidebar",
            ),

            # ── RIGHT MAIN PANEL ─────────────────────────────────────────────
            ui.div(

                # Dynamic stats row (changes with mode)
                ui.output_ui("vr_stats_row"),

                # Charts area
                ui.div(
                    # Left column: 3 small charts
                    ui.div(
                        ui.div(
                            ui.div(ui.output_text("vr_title1", inline=True),
                                   class_="card-title"),
                            ui.output_ui("vr_chart1"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div(ui.output_text("vr_title2", inline=True),
                                   class_="card-title"),
                            ui.output_ui("vr_chart2"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div(ui.output_text("vr_title3", inline=True),
                                   class_="card-title"),
                            ui.output_ui("vr_chart3"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: tall p-value histogram
                    ui.div(
                        ui.div(
                            ui.div("p-VALUE HISTOGRAM", class_="card-title"),
                            ui.output_ui("vr_pvalue_plot"),
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
