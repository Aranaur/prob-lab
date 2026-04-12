# =============================================================================
# Bootstrap Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def boot_panel() -> ui.Tag:
    return ui.nav_panel(
        "Bootstrap Explorer",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconception: "),
                    "\u201cBootstrap always works regardless of sample size.\u201d",
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "With very small n or heavy tails, bootstrap distributions "
                    "can be unstable. BCa partially corrects for bias and "
                    "skewness but cannot rescue fundamentally insufficient data.",
                    class_="info-banner-text",
                ),

                # Distribution
                ui.input_select(
                    "boot_dist",
                    ui.TagList(
                        "Distribution",
                        tip("The population from which the original sample "
                            "is drawn. Skewed and heavy-tailed cases "
                            "highlight where BCa outperforms simpler CIs."),
                    ),
                    choices={
                        "normal":    "Normal  (symmetric)",
                        "lognormal": "Log-normal  (skewed)",
                        "heavy":     "Heavy-tailed  (t, df\u2009=\u20093)",
                        "uniform":   "Uniform",
                        "bimodal":   "Bimodal  (mixture)",
                    },
                    selected="normal", width="100%",
                ),

                # Statistic
                ui.input_select(
                    "boot_statistic",
                    ui.TagList(
                        "Statistic",
                        tip("The sample statistic to bootstrap. "
                            "Studentized CI is only available for the mean "
                            "(requires analytic SE formula)."),
                    ),
                    choices={
                        "mean":    "Mean",
                        "median":  "Median",
                        "trimmed": "Trimmed Mean (10%)",
                        "percentile90": "90th Percentile",
                        "std":     "Std Deviation",
                    },
                    selected="mean", width="100%",
                ),

                # n and B
                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "boot_n",
                            ui.TagList("n (sample size)",
                                       tip("Number of observations in the "
                                           "original sample.")),
                            value=30, min=5, max=500, step=5, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric(
                            "boot_B",
                            ui.TagList("B (resamples)",
                                       tip("Number of bootstrap resamples "
                                           "per experiment.")),
                            value=2000, min=200, max=10000, step=100,
                            width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),

                # Confidence level
                ui.input_slider(
                    "boot_conf",
                    ui.TagList("Confidence level (%)",
                               tip("Nominal coverage probability for CIs.")),
                    min=80, max=99, value=95, step=1, width="100%",
                ),

                # CI methods (dynamic — Studentized only for mean)
                ui.output_ui("boot_ci_methods_ui"),

                # Step-by-step toggle
                ui.input_switch(
                    "boot_step_mode",
                    ui.TagList(
                        "Step-by-step mode",
                        tip("Animate individual resamples so you can watch "
                            "the bootstrap distribution build up. "
                            "\u00d71\u2009=\u20091 resample, "
                            "\u00d7100\u2009=\u2009complete experiment."),
                    ),
                    value=False,
                ),

                # Presets
                ui.tags.label(
                    "Presets",
                    style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
                ),
                ui.div(
                    ui.input_action_button("boot_pre_skewed", "Skewed+Med",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("boot_pre_heavy",  "Heavy tails",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("boot_pre_bimodal", "Bimodal",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("boot_pre_tiny",   "n\u2009=\u20095",
                                           class_="btn-ctrl btn-preset"),
                    class_="np-preset-grid",
                ),
                ui.output_ui("boot_preset_desc"),

                # ── Sampling controls ────────────────────────────────────────
                ui.div(
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("boot_btn_sample_1", "\u00d71",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("boot_btn_sample_50", "\u00d750",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("boot_btn_sample_100", "\u00d7100",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("boot_speed_minus", "\u2212",
                                                   class_="btn-ctrl btn-pm"),
                            ui.input_action_button("boot_btn_play", "Play",
                                                   class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("boot_speed_plus", "+",
                                                   class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.input_action_button("boot_btn_reset", "Reset",
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
                ui.navset_underline(
                    # ── Tab A: Mechanics & Coverage ──────────────────────
                    ui.nav_panel(
                        "Mechanics & Coverage",

                        # Stats row (dynamic)
                        ui.output_ui("boot_stats_row"),

                        # Charts area
                        ui.div(
                            # Left column: 3 charts
                            ui.div(
                                ui.div(
                                    ui.div("ORIGINAL SAMPLE + RESAMPLE",
                                           class_="card-title"),
                                    ui.output_ui("boot_sample_plot"),
                                    class_="glass-card chart-card",
                                ),
                                ui.div(
                                    ui.div("BOOTSTRAP DISTRIBUTION",
                                           class_="card-title"),
                                    ui.output_ui("boot_dist_plot"),
                                    class_="glass-card chart-card",
                                ),
                                ui.div(
                                    ui.div("CI COMPARISON",
                                           class_="card-title"),
                                    ui.output_ui("boot_ci_plot"),
                                    class_="glass-card chart-card",
                                ),
                                class_="charts-col-left",
                            ),
                            # Right column: tall coverage chart
                            ui.div(
                                ui.div(
                                    ui.div("COVERAGE SIMULATION",
                                           class_="card-title"),
                                    ui.output_ui("boot_coverage_plot"),
                                    class_="glass-card chart-card",
                                ),
                                class_="charts-col-right",
                            ),
                            class_="charts-area",
                        ),
                    ),

                    # ── Tab B: Convergence Analysis ──────────────────────
                    ui.nav_panel(
                        "Convergence Analysis",

                        ui.div(
                            # Info banner
                            ui.div(
                                ui.tags.span("\U0001f4ca\u00a0", class_="info-icon"),
                                ui.HTML(
                                    "<strong>What this shows:</strong> "
                                    "For each sample size N, the simulation runs <em>1\u202f000 independent experiments</em> "
                                    "and measures how often each CI method <em>fails to cover</em> the true parameter "
                                    "(False Positive Rate). "
                                    "A well-calibrated method should stay near the nominal \u03b1 line. "
                                    "Shaded bands show \u00b11.96\u00d7SE of the FPR estimate across experiments."
                                    "<br>"
                                    "<strong>Sidebar controls that matter here:</strong> "
                                    "\u2018Distribution\u2019 (shape of the data), "
                                    "\u2018Statistic\u2019 (what we estimate), "
                                    "\u2018CI Methods\u2019 (which intervals to compare), "
                                    "and \u2018Conf. level\u2019 (\u03b1 reference line). "
                                    "<em>Sample size (n) and B are fixed internally: "
                                    "N \u2208 {5,\u200910,\u200920,\u200930,\u200950,\u2009100,\u2009200,\u2009500}, B\u2009=\u2009400.</em>"
                                ),
                                class_="info-banner-text",
                                style="margin-bottom:0.5rem; flex-shrink:0;",
                            ),
                            # Controls row
                            ui.div(
                                ui.output_ui("boot_conv_ref_ui"),
                                ui.input_task_button(
                                    "boot_conv_run",
                                    "Run Convergence Simulation",
                                    class_="btn-ctrl btn-sample",
                                ),
                                style=("display:flex; justify-content:space-between; "
                                       "align-items:flex-end; padding:0.5rem 0; "
                                       "flex-wrap:wrap; gap:0.5rem;"),
                            ),
                            # Chart
                            ui.div(
                                ui.div("FPR vs SAMPLE SIZE",
                                       class_="card-title"),
                                ui.output_ui("boot_conv_plot"),
                                class_="glass-card chart-card",
                                style="min-height:460px;",
                            ),

                            style="display:flex; flex-direction:column; height:100%;",
                        ),
                    ),
                    id="boot_main_tabs",
                ),
                class_="main-panel",
            ),

            class_="app-body",
        ),
    )
