# =============================================================================
# Power Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def power_panel() -> ui.Tag:
    return ui.nav_panel(
        "Power Explorer",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconception: "),
                    "A non-significant result means the effect doesn\u2019t exist.",
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "It may simply mean the study lacked ",
                    ui.tags.em("statistical power"),
                    " to detect the effect.",
                    class_="info-banner-text",
                ),

                # Metric type toggle
                ui.input_select(
                    "pw_metric_type",
                    ui.TagList(
                        "Metric type",
                        tip("Continuous: differences in means (Cohen\u2019s d). "
                            "Proportions: differences in rates (p\u2080 vs p\u2081). "
                            "Ratio: ratio metrics X/Y via Delta Method."),
                    ),
                    choices={
                        "continuous": "Continuous (means)",
                        "proportion": "Proportions (rates)",
                        "ratio":      "Ratio (delta method)",
                    },
                    selected="continuous", width="100%",
                ),

                # Solve for
                ui.input_select(
                    "pw_solve_for",
                    ui.TagList(
                        "Solve for",
                        tip("Fix three parameters and compute the fourth. "
                            "The input for the computed parameter is replaced "
                            "with its calculated value."),
                    ),
                    choices={
                        "power": "Power (1\u2212\u03b2)",
                        "n":     "Sample size (n)",
                        "d":     "Effect size (d)",
                        "alpha": "Significance level (\u03b1)",
                    },
                    selected="power", width="100%",
                ),

                # Computed result — shown right after "Solve for"
                ui.output_ui("pw_computed_result"),

                # Test type — dynamic choices per metric type
                ui.output_ui("pw_test_type_ui"),

                # Alternative
                ui.input_select(
                    "pw_alternative",
                    ui.TagList("Alternative", tip("Direction of the alternative hypothesis.")),
                    choices={
                        "two-sided": "Two-sided (\u2260)",
                        "greater":   "Right-tailed (>)",
                        "less":      "Left-tailed (<)",
                    },
                    selected="two-sided", width="100%",
                ),

                # Effect size d (continuous) or p₀/p₁ (proportion) — dynamic
                ui.output_ui("pw_input_d"),

                # Sample size — input or hidden when computed
                ui.output_ui("pw_input_n"),

                # Alpha — input or hidden when computed
                ui.output_ui("pw_input_alpha"),

                # Power — input or hidden when computed
                ui.output_ui("pw_input_power"),


                # Presets — dynamic per metric type
                ui.output_ui("pw_presets_ui"),
                ui.output_ui("pw_preset_desc"),

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

                # Stats row — dynamic per metric type
                ui.output_ui("pw_stats_row"),

                # Charts — left: stacked distributions; right: power curve
                ui.div(
                    # Left column: sampling distributions + effect overlap
                    ui.div(
                        ui.div(
                            ui.div("H\u2080 / H\u2081 SAMPLING DISTRIBUTIONS", class_="card-title"),
                            ui.output_ui("pw_dist_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("EFFECT SIZE VISUALISATION", class_="card-title"),
                            ui.output_ui("pw_overlap_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: power curve
                    ui.div(
                        ui.div(
                            ui.div("POWER CURVE", class_="card-title"),
                            ui.output_ui("pw_curve_plot"),
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
