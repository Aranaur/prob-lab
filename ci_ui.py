from shiny import ui
from utils import tip


def ci_panel() -> ui.Tag:
    return ui.nav_panel(
        "CI Explorer",

        ui.div(

            # ── LEFT SIDEBAR: controls ────────────────────────────────────
            ui.div(

                # Interpretation panel — Correct vs Incorrect framing
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Interpreting a Confidence Interval"),
                    ui.tags.br(),
                    ui.tags.strong("Correct: "),
                    ui.output_text("ci_conf_pct", inline=True),
                    " of intervals cover \u03b8 over repeated samples.",
                    ui.tags.br(),
                    ui.tags.strong("Incorrect: "),
                    "\u201cThis interval has ",
                    ui.output_text("ci_conf_pct2", inline=True),
                    " probability of containing \u03b8.\u201d",
                    ui.tags.br(),
                    ui.tags.em(
                        "Any single interval either contains \u03b8 or not \u2014 the ",
                        ui.output_text("ci_conf_pct3", inline=True),
                        " refers to the procedure, not this specific interval.",
                    ),
                    class_="info-banner-text",
                ),

                # Confidence level slider
                ui.input_slider(
                    "ci_conf_level",
                    ui.TagList("Confidence Level (%)", tip("The probability that the interval estimation procedure will produce an interval containing the true parameter.")),
                    min=50, max=99, value=95, step=1, width="100%",
                ),

                # Distribution selector
                ui.input_select(
                    "ci_pop_dist",
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
                ui.output_ui("ci_dynamic_params"),

                # Statistic selector (dynamic — adds Proportion for Binomial)
                ui.output_ui("ci_statistic_ui"),

                # Percentile level — shown only when Percentile is selected
                ui.output_ui("ci_percentile_param"),

                # CI method selector
                ui.input_select(
                    "ci_method",
                    ui.TagList("CI Method", tip(
                        "t-interval: uses sample s, assumes normal sampling distribution. "
                        "z-interval: uses true population \u03c3 (known) \u2014 rarely realistic. "
                        "Bootstrap: resamples from the observed data \u2014 no distributional assumptions."
                    )),
                    choices={
                        "t":         "t-interval  (unknown \u03c3)",
                        "z":         "z-interval  (known \u03c3) \u26a0 rarely realistic",
                        "bootstrap": "Bootstrap   (percentile, B\u200a=\u200a500)",
                    },
                    selected="t",
                    width="100%",
                ),

                # Bootstrap-method note (points to the richer Bootstrap Explorer)
                ui.div(
                    "Advanced bootstrap methods (BCa, Studentized) \u2192 Bootstrap Explorer",
                    style=("font-size:0.72rem; color:var(--c-text3); "
                           "font-style:italic; margin:-6px 0 8px 2px;"),
                ),

                # Scenario presets
                ui.tags.label(
                    "Scenario presets",
                    style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
                ),
                ui.div(
                    ui.input_action_button("ci_pre_ideal",       "Ideal",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("ci_pre_skewed",      "Skewed",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("ci_pre_heavy",       "Heavy skew",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("ci_pre_rare",        "Rare events",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("ci_pre_boot",        "Bootstrap",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("ci_pre_poisson",     "Poisson",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("ci_pre_false_conf",  "False conf.",
                                           class_="btn-ctrl btn-preset"),
                    class_="np-preset-grid",
                ),
                ui.output_ui("ci_preset_desc"),

                # Sampling controls
                ui.div(
                    # Row 1: label + 3 equal sample buttons
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("ci_btn_sample_1",   "\u00d71",   class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("ci_btn_sample_50",  "\u00d750",  class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("ci_btn_sample_100", "\u00d7100", class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    # Row 3: speed + play
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("ci_speed_minus", "\u2212", class_="btn-ctrl btn-pm"),
                            ui.input_action_button("ci_btn_play", "Play", class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("ci_speed_plus", "+", class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    # Row 4: reset
                    ui.div(
                        ui.input_action_button("ci_btn_reset", "Reset", class_="btn-ctrl btn-reset btn-full"),
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
                        ui.div("CI COVERAGE\u00a0", tip("Percentage of all generated CIs that contain the true \u03bc."), class_="stat-label"),
                        ui.div(ui.output_text("ci_cov_rate", inline=True), class_="stat-value coverage"),
                        ui.output_ui("ci_cov_verdict"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            ui.output_text("ci_stat_label_inc", inline=True), " ",
                            tip("Count of intervals where the true parameter falls inside the CI."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("ci_num_covered", inline=True), class_="stat-value included"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            ui.output_text("ci_stat_label_miss", inline=True), " ",
                            tip("Count of intervals where the true parameter falls outside the CI."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("ci_num_missed", inline=True), class_="stat-value missed"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div("TOTAL EXPERIMENTS\u00a0", tip("Total number of random samples generated so far."), class_="stat-label"),
                        ui.div(ui.output_text("ci_num_total", inline=True), class_="stat-value total"),
                        class_="stat-card",
                    ),
                    class_="stats-row",
                ),

                # Charts area
                ui.div(
                    # Left column: 3 small charts (CLT on top)
                    ui.div(
                        ui.div(
                            ui.div(ui.output_text("ci_stat_plot_title", inline=True), class_="card-title"),
                            ui.output_ui("ci_means_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div(ui.output_text("ci_prop_plot_title", inline=True), class_="card-title"),
                            ui.output_ui("ci_prop_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("CI WIDTH DISTRIBUTION", class_="card-title"),
                            ui.div(
                                "Narrower intervals are not always better \u2014 "
                                "they may undercover the true value.",
                                style=("font-size:0.72rem; color:var(--c-text3); "
                                       "font-style:italic; margin:-4px 12px 4px;"),
                            ),
                            ui.output_ui("ci_width_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: population distribution + CI chart
                    ui.div(
                        ui.div(
                            ui.div("POPULATION DISTRIBUTION", class_="card-title"),
                            ui.output_ui("ci_population_plot"),
                            class_="glass-card chart-card chart-card-pop",
                        ),
                        ui.div(
                            ui.div("CONFIDENCE INTERVALS", class_="card-title"),
                            ui.div(
                                "Each interval either contains \u03b8 or not \u2014 "
                                "the confidence level describes the procedure across many samples.",
                                style=("font-size:0.72rem; color:var(--c-text3); "
                                       "font-style:italic; margin:-4px 12px 4px;"),
                            ),
                            ui.output_ui("ci_plot"),
                            ui.output_ui("ci_decision"),
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
    )
