# =============================================================================
# Nonparametric Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def np_panel() -> ui.Tag:
    return ui.nav_panel(
        "Nonparametric Explorer",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconceptions: "),
                    "Mann-Whitney U is NOT a test of medians \u2014 it tests "
                    "stochastic dominance: P(X\u2009>\u2009Y)\u2009=\u20090.5.",
                    ui.tags.br(),
                    "MW-U is NOT simply a \u201cnonparametric t-test\u201d \u2014 they test ",
                    ui.tags.em("different hypotheses."),
                    class_="info-banner-text",
                ),

                # Mode selector
                ui.input_select(
                    "np_mode",
                    ui.TagList(
                        "Mode",
                        tip(
                            "Independent: Mann-Whitney U vs Welch t-test on two independent samples. "
                            "Paired: Wilcoxon signed-rank vs paired t-test on differences. "
                            "Note: Wilcoxon signed-rank assumes symmetric differences under H\u2080."
                        ),
                    ),
                    choices={
                        "independent": "Independent  (MW-U vs t-test)",
                        "paired":      "Paired  (Wilcoxon vs paired t)",
                    },
                    selected="independent",
                    width="100%",
                ),

                # Scenario presets
                ui.tags.label(
                    "Presets",
                    style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
                ),
                ui.div(
                    ui.input_action_button("np_preset_normal_h0", "Normal H\u2080",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("np_preset_normal_h1", "Normal H\u2081",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("np_preset_outlier", "Outliers",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("np_preset_skewed", "Skewed",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("np_preset_myth", "Shape myth",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("np_preset_cauchy", "Cauchy",
                                           class_="btn-ctrl btn-preset"),
                    class_="np-preset-grid",
                ),
                ui.output_ui("np_preset_desc"),

                # Dynamic distribution selectors + parameters
                ui.output_ui("np_dist_section"),

                # Location shift
                ui.input_numeric(
                    "np_delta",
                    ui.TagList(
                        "Location shift (\u03b4)",
                        tip(
                            "Independent: added to every Group B observation. "
                            "Paired: true mean of differences. "
                            "Set to 0 to simulate Type\u00a0I error under H\u2080."
                        ),
                    ),
                    value=0.0, step=0.25, width="100%",
                ),

                # Sample size
                ui.input_numeric(
                    "np_n",
                    ui.TagList(
                        "Sample size (n)",
                        tip("Observations per group (independent) or number of pairs (paired)."),
                    ),
                    value=30, min=5, max=500, step=1, width="100%",
                ),

                # Significance level
                ui.input_slider(
                    "np_alpha",
                    ui.TagList(
                        "Significance level (\u03b1)",
                        tip("Threshold for rejecting H\u2080."),
                    ),
                    min=0.01, max=0.20, value=0.05, step=0.01, width="100%",
                ),

                # ── Sampling controls ────────────────────────────────────────
                ui.div(
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("np_btn_sample_1",   "\u00d71",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("np_btn_sample_50",  "\u00d750",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("np_btn_sample_100", "\u00d7100",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("np_speed_minus", "\u2212",
                                                   class_="btn-ctrl btn-pm"),
                            ui.input_action_button("np_btn_play", "Play",
                                                   class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("np_speed_plus", "+",
                                                   class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.input_action_button("np_btn_reset", "Reset",
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
                            ui.output_text("np_param_stat_label", inline=True), "\u00a0",
                            tip("Empirical reject rate for the parametric test."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("np_param_rate", inline=True),
                               class_="stat-value coverage"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            ui.output_text("np_nonparam_stat_label", inline=True), "\u00a0",
                            tip("Empirical reject rate for the nonparametric test."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("np_nonparam_rate", inline=True),
                               class_="stat-value np-nonparam"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "TOTAL TOTAL EXPERIMENTS\u00a0",
                            tip("Total number of simulated experiments."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("np_total_tests", inline=True),
                               class_="stat-value total"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "ADVANTAGE\u00a0",
                            tip(
                                "Nonparametric rate \u2212 Parametric rate. "
                                "Positive \u2192 nonparametric detects more. "
                                "Under H\u2080, closer to 0 means better Type\u00a0I control."
                            ),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("np_advantage", inline=True),
                               class_="stat-value np-advantage"),
                        class_="stat-card",
                    ),
                    class_="stats-row",
                ),

                # Charts area
                ui.div(
                    # Left column: 3 small charts
                    ui.div(
                        ui.div(
                            ui.div("SAMPLE DISTRIBUTIONS", class_="card-title"),
                            ui.output_ui("np_sample_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("REJECT RATE\u00a0COMPARISON", class_="card-title"),
                            ui.output_ui("np_reject_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("RANK VISUALIZATION", class_="card-title"),
                            ui.output_ui("np_rank_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: main chart
                    ui.div(
                        ui.div(
                            ui.div("p-VALUE DISTRIBUTION COMPARISON", class_="card-title"),
                            ui.output_ui("np_pvalue_plot"),
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
