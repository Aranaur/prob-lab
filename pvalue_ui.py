# =============================================================================
# p-value Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def pvalue_panel() -> ui.Tag:
    return ui.nav_panel(
        "p-value Explorer",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconception: "),
                    'p-value is NOT the probability that H\u2080 is true.',
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "It is the probability of observing a result as extreme as yours, ",
                    ui.tags.em("assuming H\u2080 is true."),
                    class_="info-banner-text",
                ),

                # ── Mode selector ─────────────────────────────────────────────
                ui.input_select(
                    "pv_mode",
                    ui.TagList(
                        "Simulation mode",
                        tip(
                            "Single experiment: the true value is fixed for every simulated test. "
                            "Shows properties of the test (Type\u00a0I error, power). "
                            "Experiment pipeline: each test is drawn from H\u2080 with probability \u03c0 "
                            "(no real effect) or from H\u2081 with probability 1\u2212\u03c0 (effect = True value). "
                            "Shows False Positive Risk \u2014 P(H\u2080 | reject) \u2014 the A/B-testing reality."
                        ),
                    ),
                    choices={
                        "single":   "Single experiment",
                        "pipeline": "Experiment pipeline (\u03c0, FPR)",
                    },
                    selected="single",
                    width="100%",
                ),
                ui.output_ui("pv_pi_control"),

                # ── Hypothesis specification ──────────────────────────────────

                # Test structure
                ui.input_select(
                    "pv_test_structure",
                    ui.TagList(
                        "Test structure",
                        tip(
                            "One-sample: tests whether a single population mean equals \u03bc\u2080. "
                            "Two-sample (independent): tests whether two independent group means differ. "
                            "Paired: tests whether the mean of within-pair differences equals \u03bc\u2080."
                        ),
                    ),
                    choices={
                        "one":    "One-sample",
                        "two":    "Two-sample (independent)",
                        "paired": "Paired",
                    },
                    selected="one",
                    width="100%",
                ),

                # Null hypothesis value
                ui.input_numeric(
                    "pv_mu0",
                    ui.TagList(
                        "H\u2080 value (\u03bc\u2080)",
                        tip(
                            "One-sample / Paired: hypothesised population mean (or mean difference). "
                            "Two-sample: hypothesised difference \u03bc\u2081\u2212\u03bc\u2082 "
                            "(usually 0)."
                        ),
                    ),
                    value=0.0, step=0.5, width="100%",
                ),

                # Alternative hypothesis — placed next to H₀ (both define the test)
                ui.input_select(
                    "pv_alternative",
                    ui.TagList(
                        "Alternative hypothesis",
                        tip(
                            "two-sided: parameter \u2260 \u03bc\u2080 | "
                            "greater: parameter > \u03bc\u2080 | "
                            "less: parameter < \u03bc\u2080"
                        ),
                    ),
                    choices={
                        "two-sided": "Two-sided  (\u2260)",
                        "greater":   "Right-tailed  (>)",
                        "less":      "Left-tailed  (<)",
                    },
                    selected="two-sided",
                    width="100%",
                ),

                # True mean / true difference — simulation ground truth
                ui.input_numeric(
                    "pv_mu_true",
                    ui.TagList(
                        "True value",
                        tip(
                            "One-sample: true population mean. "
                            "Two-sample: true \u03bc\u2081\u2212\u03bc\u2082. "
                            "Paired: true mean of differences. "
                            "Set equal to H\u2080 value to simulate Type\u00a0I error."
                        ),
                    ),
                    value=0.5, step=0.25, width="100%",
                ),

                # ── Population parameters (group-aware block) ─────────────────
                # For one-sample: single σ input.
                # For two-sample/paired: two-column σ A / σ B (+ n A / n B for two-sample).
                ui.output_ui("pv_group_params"),

                # ── Test settings ─────────────────────────────────────────────

                # Significance level
                ui.input_slider(
                    "pv_alpha",
                    ui.TagList(
                        "Significance level (\u03b1)",
                        tip(
                            "Threshold for rejecting H\u2080. "
                            "Also the long-run Type\u00a0I error rate when H\u2080 is true."
                        ),
                    ),
                    min=0.01, max=0.20, value=0.05, step=0.01, width="100%",
                ),

                # Test method
                ui.input_select(
                    "pv_test_method",
                    ui.TagList(
                        "Test method",
                        tip(
                            "t-test: \u03c3 is estimated from each sample \u2014 "
                            "the null distribution is t(df). "
                            "Realistic scenario (unknown \u03c3). "
                            "z-test: uses the true Population \u03c3 directly \u2014 "
                            "the null distribution is N(0,\u00a01). "
                            "Idealized / theoretical scenario."
                        ),
                    ),
                    choices={
                        "t": "t-test  (estimate \u03c3 from sample)",
                        "z": "z-test  (use true \u03c3)",
                    },
                    selected="t",
                    width="100%",
                ),

                # Scenario presets
                ui.tags.label(
                    "Scenario presets",
                    style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
                ),
                ui.div(
                    ui.input_action_button("pv_pre_h0",      "H\u2080 true",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pv_pre_under",   "Low power",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pv_pre_largen",  "Large n",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pv_pre_outlier", "Outlier",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pv_pre_paired",  "Paired",
                                           class_="btn-ctrl btn-preset"),
                    class_="np-preset-grid",
                ),
                ui.output_ui("pv_preset_desc"),

                # ── Robustness options ────────────────────────────────────────
                ui.tags.hr(style="border-color: rgba(255,255,255,0.12); margin: 6px 0;"),
                ui.div(
                    ui.input_checkbox(
                        "pv_outlier_on",
                        ui.TagList(
                            "Inject outlier\u00a0",
                            tip(
                                "Replaces one observation in every sample with an extreme value "
                                "opposing the true effect. "
                                "Shows how a single outlier inflates variance and pulls the "
                                "sample mean toward H\u2080, pushing a significant result "
                                "into non-significance. Larger magnitude \u2192 more broken test."
                            ),
                        ),
                        value=False,
                    ),
                    ui.input_checkbox(
                        "pv_wilcoxon_on",
                        ui.TagList(
                            "Wilcoxon test\u00a0",
                            tip(
                                "Runs a nonparametric test alongside the t/z-test on every sample. "
                                "One-sample & Paired \u2192 Wilcoxon signed-rank; "
                                "Two-sample \u2192 Mann-Whitney U. "
                                "Especially useful with outlier injection to see how "
                                "nonparametric tests resist contamination."
                            ),
                        ),
                        value=False,
                    ),
                    class_="pv-checks-row",
                ),
                ui.output_ui("pv_outlier_slider"),

                # ── Cross-reference: Sequential Testing ───────────────────────
                ui.div(
                    ui.tags.strong("See also: "),
                    "curious how checking p-values every day inflates false positives? "
                    "The ",
                    ui.tags.em("Sequential Testing"),
                    " tab covers peeking, O\u2019Brien\u2013Fleming, Pocock, and Haybittle\u2013Peto boundaries.",
                    class_="info-banner-text",
                    style="margin: 6px 0;",
                ),

                # ── Sampling controls ─────────────────────────────────────────
                ui.div(
                    # Row 1: sample size (hidden for two-sample — n lives in group params)
                    ui.output_ui("pv_n_control"),
                    # Row 2: sample buttons
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("pv_btn_sample_1",   "\u00d71",   class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("pv_btn_sample_50",  "\u00d750",  class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("pv_btn_sample_100", "\u00d7100", class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    # Row 3: speed + play
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("pv_speed_minus", "\u2212", class_="btn-ctrl btn-pm"),
                            ui.input_action_button("pv_btn_play", "Play", class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("pv_speed_plus", "+", class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    # Row 4: reset
                    ui.div(
                        ui.input_action_button("pv_btn_reset", "Reset", class_="btn-ctrl btn-reset btn-full"),
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
                            "CURRENT p-VALUE\u00a0",
                            tip("p-value of the most recently drawn sample."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pv_current_pvalue", inline=True), class_="stat-value coverage"),
                        class_="stat-card",
                    ),
                    ui.output_ui("pv_reject_stat_card"),
                    ui.output_ui("pv_fpr_stat_card"),
                    ui.div(
                        ui.div(
                            "TOTAL EXPERIMENTS\u00a0",
                            tip("Total number of hypothesis tests simulated."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pv_total_tests", inline=True), class_="stat-value total"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "THEORETICAL POWER\u00a0",
                            tip(
                                "Probability of correctly rejecting H\u2080 "
                                "given the current parameters."
                            ),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("pv_theo_power", inline=True), class_="stat-value missed"),
                        class_="stat-card",
                    ),
                    class_="stats-row",
                ),

                # Charts area
                ui.div(
                    # Left column: p-value histogram + power diagram
                    ui.div(
                        ui.div(
                            ui.div("p-VALUE DISTRIBUTION", class_="card-title"),
                            ui.output_ui("pv_hist_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("POWER DIAGRAM", class_="card-title"),
                            ui.output_ui("pv_power_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: null distribution + effect-size inflation
                    ui.div(
                        ui.div(
                            ui.div("NULL DISTRIBUTION & TEST STATISTIC", class_="card-title"),
                            ui.output_ui("pv_null_dist_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div(
                                "EFFECT SIZE INFLATION (WINNER'S CURSE)\u00a0",
                                tip(
                                    "Each dot is one simulation. Purple = significant (p < \u03b1), "
                                    "grey = non-significant. "
                                    "The solid blue line is the true effect; the dashed purple line "
                                    "is the average observed effect among significant results. "
                                    "Their gap is the Winner\u2019s Curse \u2014 under low power, "
                                    "only the lucky extremes cross the \u03b1 threshold, so published "
                                    "significant effects exaggerate reality."
                                ),
                                class_="card-title",
                            ),
                            ui.output_ui("pv_effect_scatter_plot"),
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
