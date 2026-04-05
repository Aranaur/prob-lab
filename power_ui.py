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

                # Test type
                ui.input_select(
                    "pw_test_type",
                    ui.TagList(
                        "Test type",
                        tip("Z-test uses known \u03c3. "
                            "t-tests estimate \u03c3 from data, giving slightly less power."),
                    ),
                    choices={
                        "one_z":    "One-sample Z",
                        "one_t":    "One-sample t",
                        "two_t":    "Two-sample t (independent)",
                        "paired_t": "Paired t",
                    },
                    selected="one_t", width="100%",
                ),

                # Alternative
                ui.input_select(
                    "pw_alternative",
                    ui.TagList("Alternative", tip("Direction of the alternative hypothesis.")),
                    choices={
                        "two-sided": "Two-sided",
                        "greater":   "Greater (right-tailed)",
                        "less":      "Less (left-tailed)",
                    },
                    selected="two-sided", width="100%",
                ),

                # Effect size d — input or computed display
                ui.output_ui("pw_input_d"),

                # Sample size — input or computed display
                ui.output_ui("pw_input_n"),

                # Alpha — input or computed display
                ui.output_ui("pw_input_alpha"),

                # Power — input or computed display
                ui.output_ui("pw_input_power"),

                # Dynamic params (n₂ for two-sample)
                ui.output_ui("pw_dynamic_params"),

                # Presets
                ui.input_select(
                    "pw_preset",
                    ui.TagList("Scenario presets",
                               tip("Load typical parameter sets for common study designs.")),
                    choices={
                        "":         "\u2014 Custom \u2014",
                        "clinical": "Clinical trial  (d\u200a=\u200a0.3, \u03b1\u200a=\u200a0.01)",
                        "ab_test":  "A/B test  (d\u200a=\u200a0.2, two-sample)",
                        "psych":    "Psychology  (d\u200a=\u200a0.5, \u03b1\u200a=\u200a0.05)",
                        "small":    "Small effect  (d\u200a=\u200a0.2, power\u200a=\u200a0.8)",
                    },
                    selected="", width="100%",
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
                            "EFFECT SIZE (d) ",
                            tip("Cohen\u2019s d: the standardised distance between H\u2080 and H\u2081."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pw_stat_d", inline=True), class_="stat-value pw-d"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "SAMPLE SIZE (n) ",
                            tip("Number of observations per group."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pw_stat_n", inline=True), class_="stat-value pw-n"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "\u03b1 (TYPE I) ",
                            tip("Probability of rejecting H\u2080 when it is true."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pw_stat_alpha", inline=True), class_="stat-value pw-alpha"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "POWER (1\u2212\u03b2) ",
                            tip("Probability of rejecting H\u2080 when H\u2081 is true."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pw_stat_power", inline=True), class_="stat-value pw-power"),
                        class_="stat-card",
                    ),
                    class_="stats-row",
                ),

                # Charts
                ui.div(
                    ui.div(
                        ui.div("H\u2080 / H\u2081 SAMPLING DISTRIBUTIONS", class_="card-title"),
                        ui.output_ui("pw_dist_plot"),
                        class_="glass-card chart-card",
                    ),
                    ui.div(
                        ui.div("POWER CURVE", class_="card-title"),
                        ui.output_ui("pw_curve_plot"),
                        class_="glass-card chart-card",
                    ),
                    class_="charts-area pw-charts",
                ),

                class_="main-panel",
            ),

            class_="app-body",
        ),
    )
