# =============================================================================
# Peeking & Sequential Testing Explorer — UI panel
# =============================================================================

from shiny import ui
from utils import tip


def seq_panel() -> ui.Tag:
    return ui.nav_panel(
        "Sequential Testing",

        ui.div(

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            ui.div(

                # Misconception banner
                ui.div(
                    ui.tags.i(class_="info-icon"),
                    ui.tags.strong(" Common Misconception: "),
                    "\u201cI can peek at p-values while collecting data and stop "
                    "when p\u2009<\u20090.05.\u201d",
                    ui.tags.br(),
                    ui.tags.strong("Reality: "),
                    "Under H\u2080 a running p-value is a random walk that will "
                    "cross ",
                    ui.tags.em("any"),
                    " threshold with probability 1 given enough observations. "
                    "Sequential boundaries (OBF, Pocock) solve this by spending "
                    "\u03b1 across planned interim looks.",
                    class_="info-banner-text",
                ),

                # Mode
                ui.input_select(
                    "seq_mode",
                    ui.TagList(
                        "Mode",
                        tip(
                            "Peeking: compute p-value after every new observation "
                            "\u2014 shows how Type\u00a0I error inflates. "
                            "Sequential: compute p only at K planned interim looks "
                            "with corrected boundaries."
                        ),
                    ),
                    choices={
                        "peeking":    "Peeking  (continuous monitoring)",
                        "sequential": "Sequential  (K planned looks)",
                    },
                    selected="peeking", width="100%",
                ),

                # Sequential-only controls
                ui.output_ui("seq_k_control"),
                ui.output_ui("seq_boundary_control"),

                # Simulation parameters
                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "seq_N",
                            ui.TagList(
                                "Max N",
                                tip("Maximum sample size per experiment."),
                            ),
                            value=200, min=30, max=2000, step=10, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric(
                            "seq_n_min",
                            ui.TagList(
                                "First look n",
                                tip("Minimum observations before the first p-value is computed. "
                                    "Avoids noisy t-statistics at very small n."),
                            ),
                            value=10, min=5, max=100, step=1, width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),

                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "seq_delta",
                            ui.TagList(
                                "\u03b4 (true effect)",
                                tip("Set to 0 to simulate Type\u00a0I error (H\u2080). "
                                    "Set > 0 to see power and early stopping."),
                            ),
                            value=0.0, step=0.1, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric(
                            "seq_sigma",
                            ui.TagList("\u03c3", tip("Population standard deviation.")),
                            value=1.0, min=0.1, step=0.5, width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),

                ui.input_slider(
                    "seq_alpha",
                    ui.TagList(
                        "\u03b1 (nominal)",
                        tip("Nominal significance level. Peeking inflates the "
                            "actual FWER far above this value."),
                    ),
                    min=0.01, max=0.20, value=0.05, step=0.01, width="100%",
                ),

                ui.input_slider(
                    "seq_n_traj",
                    ui.TagList(
                        "Trajectories shown",
                        tip("Number of previous experiment trajectories "
                            "displayed as faint background lines."),
                    ),
                    min=5, max=100, value=20, step=5, width="100%",
                ),

                # ── Sampling controls ────────────────────────────────────────
                ui.div(
                    ui.div(
                        ui.tags.span("Sample:", class_="btn-row-label"),
                        ui.input_action_button("seq_btn_sample_1", "\u00d71",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("seq_btn_sample_50", "\u00d750",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        ui.input_action_button("seq_btn_sample_100", "\u00d7100",
                                               class_="btn-ctrl btn-sample btn-flex"),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.div(
                            ui.tags.label("Speed"),
                            ui.input_action_button("seq_speed_minus", "\u2212",
                                                   class_="btn-ctrl btn-pm"),
                            ui.input_action_button("seq_btn_play", "Play",
                                                   class_="btn-ctrl btn-play btn-flex"),
                            ui.input_action_button("seq_speed_plus", "+",
                                                   class_="btn-ctrl btn-pm"),
                            class_="ctrl-group ctrl-group-full",
                        ),
                        class_="sidebar-btn-row",
                    ),
                    ui.div(
                        ui.input_action_button("seq_btn_reset", "Reset",
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
                            ui.output_text("seq_lbl_peek", inline=True), "\u00a0",
                            tip("Empirical rejection rate when peeking after every observation."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("seq_peek_rate", inline=True),
                               class_="stat-value missed"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            ui.output_text("seq_lbl_seq", inline=True), "\u00a0",
                            tip("Empirical rejection rate with sequential boundaries."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("seq_seq_rate", inline=True),
                               class_="stat-value coverage"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "AVG STOP n\u00a0",
                            tip("Average sample size at which sequential experiments stop "
                                "(either by crossing the boundary or reaching N)."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("seq_avg_stop", inline=True),
                               class_="stat-value included"),
                        class_="stat-card",
                    ),
                    ui.div(
                        ui.div(
                            "EXPERIMENTS\u00a0",
                            tip("Total simulated experiments."),
                            class_="stat-label",
                        ),
                        ui.div(ui.output_text("seq_total_val", inline=True),
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
                            ui.div("SEQUENTIAL BOUNDARIES (z-scale)",
                                   class_="card-title"),
                            ui.output_ui("seq_boundary_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div("STOPPING DISTRIBUTION", class_="card-title"),
                            ui.output_ui("seq_stop_plot"),
                            class_="glass-card chart-card",
                        ),
                        ui.div(
                            ui.div(ui.output_text("seq_title_error", inline=True),
                                   class_="card-title"),
                            ui.output_ui("seq_error_plot"),
                            class_="glass-card chart-card",
                        ),
                        class_="charts-col-left",
                    ),
                    # Right column: main trajectory chart
                    ui.div(
                        ui.div(
                            ui.div("p-VALUE TRAJECTORY", class_="card-title"),
                            ui.output_ui("seq_traj_plot"),
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
