# =============================================================================
# Multiple Testing Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def mt_panel() -> ui.Tag:
    return ui.nav_panel(
        "Multiple Testing",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconception: "),
                    "Testing 20 hypotheses at \u03b1\u200a=\u200a0.05 still controls "
                    "the error at 5% per test.",
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "P(\u2265\u200a1 false positive) = 1\u2009\u2212\u2009(1\u2212\u03b1)",
                    ui.tags.sup("m"),
                    " \u2248\u00a064% for m\u200a=\u200a20. "
                    "Multiple-testing corrections are essential.",
                    class_="info-banner-text",
                ),

                # ── Simulation parameters ────────────────────────────────────
                ui.input_slider(
                    "mt_m",
                    ui.TagList(
                        "Number of hypotheses (m)",
                        tip("Total number of tests performed simultaneously. "
                            "More tests \u2192 higher FWER without correction."),
                    ),
                    min=2, max=200, value=20, step=1, width="100%",
                ),

                ui.output_ui("mt_k_control"),

                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "mt_delta",
                            ui.TagList(
                                "\u03b4 (effect size)",
                                tip("True effect for H\u2081 hypotheses. "
                                    "Set k\u200a>\u200a0 to see power vs. correction trade-off."),
                            ),
                            value=0.5, step=0.25, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric(
                            "mt_sigma",
                            ui.TagList(
                                "\u03c3",
                                tip("Population standard deviation for each test."),
                            ),
                            value=1.0, min=0.1, step=0.5, width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),

                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "mt_n",
                            ui.TagList(
                                "Sample size (n)",
                                tip("Observations per test."),
                            ),
                            value=30, min=5, max=500, step=1, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_slider(
                            "mt_alpha",
                            ui.TagList(
                                "\u03b1",
                                tip("Per-test significance level before correction."),
                            ),
                            min=0.01, max=0.20, value=0.05, step=0.01, width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),

                # ── Correlation structure ────────────────────────────────────
                ui.input_select(
                    "mt_corr_struct",
                    ui.TagList(
                        "Test dependence",
                        tip(
                            "Independent: all tests are uncorrelated. "
                            "Block: equi-correlated tests (shared latent factor). "
                            "BH controls FDR under independence; BY controls FDR "
                            "under arbitrary dependence."
                        ),
                    ),
                    choices={
                        "independent": "Independent",
                        "block":       "Block correlation (\u03c1)",
                    },
                    selected="independent", width="100%",
                ),
                ui.output_ui("mt_rho_control"),

                # ── Display options ──────────────────────────────────────────
                ui.tags.hr(style="border-color: rgba(255,255,255,0.12); margin: 6px 0;"),

                ui.input_select(
                    "mt_scatter_x",
                    ui.TagList(
                        "Scatter X-axis",
                        tip(
                            "Original: test index \u2014 shows random noise pattern. "
                            "Rank: sorted by p-value \u2014 shows BH/BY diagonal thresholds."
                        ),
                    ),
                    choices={
                        "rank":     "Rank (sorted by p)",
                        "original": "Original index",
                    },
                    selected="rank", width="100%",
                ),

                ui.input_checkbox(
                    "mt_file_drawer",
                    ui.TagList(
                        "File drawer effect\u00a0",
                        tip(
                            "Hide all p \u2265 0.05 from scatter and histogram \u2014 "
                            "simulates selective reporting / publication bias."
                        ),
                    ),
                    value=False,
                ),

                # ── Sampling controls ────────────────────────────────────────
                ui.div(
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("mt_btn_sample_1", "\u00d71",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("mt_btn_sample_50", "\u00d750",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("mt_btn_sample_100", "\u00d7100",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("mt_speed_minus", "\u2212",
                                                   class_="btn-ctrl btn-pm"),
                            ui.input_action_button("mt_btn_play", "Play",
                                                   class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("mt_speed_plus", "+",
                                                   class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.input_action_button("mt_btn_reset", "Reset",
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
                    ui.tags.a("Website", href="https://aranaur.rbind.io/",
                              target="_blank"),
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
                            "FWER (UNCORR.)\u00a0",
                            tip(
                                "Empirical family-wise error rate without correction: "
                                "fraction of experiments with \u2265\u200a1 false positive. "
                                "Theoretical: 1\u2009\u2212\u2009(1\u2212\u03b1)\u1d50."
                            ),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("mt_fwer_val", inline=True),
                               class_="stat-value missed"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "FDR (BH)\u00a0",
                            tip(
                                "Empirical false discovery rate after BH correction: "
                                "avg(FP / Discoveries) per experiment. "
                                "BH controls FDR \u2264 \u03b1 under independence."
                            ),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("mt_fdr_val", inline=True),
                               class_="stat-value coverage"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "POWER (BH)\u00a0",
                            tip(
                                "Empirical power after BH correction: "
                                "fraction of true H\u2081 hypotheses detected."
                            ),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("mt_power_val", inline=True),
                               class_="stat-value included"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "EXPERIMENTS\u00a0",
                            tip("Total number of simulated experiments."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("mt_total_val", inline=True),
                               class_="stat-value total"),
                        class_="stat-card",
                    ),
                    class_="stats-row",
                ),

                # Charts area
                ui.div(
                    # Left column: 3 small charts
                    ui.div(
                        ui.div(
                            ui.div("p-VALUE SCATTER", class_="card-title"),
                            ui.output_ui("mt_scatter_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("CORRECTION COMPARISON", class_="card-title"),
                            ui.output_ui("mt_bars_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("FWER vs NUMBER OF TESTS", class_="card-title"),
                            ui.output_ui("mt_fwer_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: BH table + histogram
                    ui.div(
                        ui.div(
                            ui.div("BH STEP-BY-STEP TABLE", class_="card-title"),
                            ui.output_ui("mt_bh_table"),
                            class_="glass-card chart-card chart-card-bh",
                        ),
                        ui.div(
                            ui.div("p-VALUE DISTRIBUTION (ALL TESTS)",
                                   class_="card-title"),
                            ui.output_ui("mt_hist_plot"),
                            class_="glass-card chart-card chart-card-phist",
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
