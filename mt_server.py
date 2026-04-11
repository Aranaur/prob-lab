# =============================================================================
# Multiple Testing Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from mt_plots import (
    draw_mt_pvalue_scatter,
    draw_mt_correction_bars,
    draw_mt_fwer_curve,
    draw_mt_pvalue_hist,
)

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
_METHODS = ["none", "bonferroni", "holm", "bh", "by"]
_METHOD_LABELS = {
    "none": "None", "bonferroni": "Bonferroni",
    "holm": "Holm", "bh": "BH", "by": "BY",
}


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


def _empty_stats():
    return {m: {"fp": 0, "tp": 0} for m in _METHODS}


def _empty_fwer():
    return {m: 0 for m in _METHODS}


def _empty_fdr():
    return {m: 0.0 for m in _METHODS}


# ── Correction algorithms ────────────────────────────────────────────────────
def _apply_corrections(pvalues: np.ndarray, alpha: float) -> dict:
    """Apply all 5 correction methods. Returns {method: rejected_mask}."""
    m = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    ranks = np.arange(1, m + 1)

    results = {}

    # Uncorrected
    results["none"] = pvalues < alpha

    # Bonferroni
    results["bonferroni"] = pvalues < (alpha / m)

    # Holm (step-down)
    holm_rej = np.zeros(m, dtype=bool)
    for i in range(m):
        if sorted_p[i] < alpha / (m - i):
            holm_rej[sorted_idx[i]] = True
        else:
            break
    results["holm"] = holm_rej

    # Benjamini-Hochberg
    bh_thresholds = ranks * alpha / m
    below = sorted_p <= bh_thresholds
    bh_rej = np.zeros(m, dtype=bool)
    if below.any():
        max_k = int(np.max(np.where(below)[0]))
        bh_rej[sorted_idx[: max_k + 1]] = True
    results["bh"] = bh_rej

    # Benjamini-Yekutieli
    c_m = float(np.sum(1.0 / ranks))
    by_thresholds = ranks * alpha / (m * c_m)
    below_by = sorted_p <= by_thresholds
    by_rej = np.zeros(m, dtype=bool)
    if below_by.any():
        max_k = int(np.max(np.where(below_by)[0]))
        by_rej[sorted_idx[: max_k + 1]] = True
    results["by"] = by_rej

    return results


# ═════════════════════════════════════════════════════════════════════════════
def mt_server(input, output, session, is_dark):

    MAX_DATA = 10_000

    # ── Reactive state ────────────────────────────────────────────────────
    mt_total = reactive.value(0)
    mt_method_stats: reactive.Value[dict] = reactive.value(_empty_stats())
    mt_fwer_counts: reactive.Value[dict] = reactive.value(_empty_fwer())
    mt_fdr_sums: reactive.Value[dict] = reactive.value(_empty_fdr())
    mt_all_pvalues: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    mt_last_pvalues: reactive.Value[np.ndarray | None] = reactive.value(None)
    mt_last_is_h1: reactive.Value[np.ndarray | None] = reactive.value(None)
    mt_is_playing = reactive.value(False)
    mt_speed_ms = reactive.value(0.5)

    # ── Safe input helpers ────────────────────────────────────────────────
    def _safe(input_id, default):
        try:
            v = getattr(input, input_id)()
            return type(default)(v) if v is not None else default
        except Exception:
            return default

    def _get_m():     return max(int(_safe("mt_m", 20)), 2)
    def _get_k():     return max(0, min(int(_safe("mt_k", 0)), _get_m()))
    def _get_delta(): return _safe("mt_delta", 0.5)
    def _get_n():     return max(int(_safe("mt_n", 30)), 5)
    def _get_sigma(): return max(_safe("mt_sigma", 1.0), 0.1)
    def _get_alpha(): return _safe("mt_alpha", 0.05)
    def _get_rho():
        try:
            s = input.mt_corr_struct()
        except Exception:
            s = "independent"
        if s != "block":
            return 0.0
        return max(0.0, min(0.95, _safe("mt_rho", 0.3)))

    # ── Dynamic k slider (max depends on m) ──────────────────────────────
    @render.ui
    def mt_k_control():
        m = _get_m()
        with reactive.isolate():
            try:
                cur = input.mt_k()
                if cur is None:
                    cur = 0
            except Exception:
                cur = 0
        k_val = min(int(cur), m)
        return ui.input_slider(
            "mt_k",
            ui.TagList(
                "True H\u2081 hypotheses (k)",
                tip(
                    "Number of hypotheses with a real effect (\u03b4 \u2260 0). "
                    "Set k=0 for a pure Garden of Forking Paths scenario (all H\u2080)."
                ),
            ),
            min=0, max=m, value=k_val, step=1, width="100%",
        )

    # ── Dynamic ρ slider (shown only for block) ──────────────────────────
    @render.ui
    def mt_rho_control():
        try:
            s = input.mt_corr_struct()
        except Exception:
            s = "independent"
        if s != "block":
            return ui.div()
        return ui.input_slider(
            "mt_rho",
            ui.TagList(
                "\u03c1 (equi-correlation)",
                tip(
                    "Correlation between every pair of tests. "
                    "Higher \u03c1 \u2192 BH loses FDR guarantee; BY maintains it."
                ),
            ),
            min=0.0, max=0.95, value=0.3, step=0.05, width="100%",
        )

    # ── Speed ± ──────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.mt_speed_minus)
    def _mt_spd_dn():
        mt_speed_ms.set(min(mt_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.mt_speed_plus)
    def _mt_spd_up():
        mt_speed_ms.set(max(mt_speed_ms() - 0.05, 0.05))

    # ── Play / Pause ─────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.mt_btn_play)
    def _mt_toggle():
        mt_is_playing.set(not mt_is_playing())
        ui.update_action_button("mt_btn_play",
                                label="Pause" if mt_is_playing() else "Play")

    @reactive.effect
    def _mt_auto():
        if mt_is_playing():
            reactive.invalidate_later(mt_speed_ms())
            with reactive.isolate():
                _draw_samples(1)

    # ── Manual buttons ───────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.mt_btn_sample_1)
    def _s1():
        _draw_samples(1)

    @reactive.effect
    @reactive.event(input.mt_btn_sample_50)
    def _s50():
        _draw_samples(50)

    @reactive.effect
    @reactive.event(input.mt_btn_sample_100)
    def _s100():
        _draw_samples(100)

    # ── Reset ────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.mt_btn_reset, input.mt_m, input.mt_alpha,
                    input.mt_corr_struct, input.mt_file_drawer)
    def _mt_reset():
        mt_total.set(0)
        mt_method_stats.set(_empty_stats())
        mt_fwer_counts.set(_empty_fwer())
        mt_fdr_sums.set(_empty_fdr())
        mt_all_pvalues.set(deque(maxlen=MAX_DATA))
        mt_last_pvalues.set(None)
        mt_last_is_h1.set(None)
        mt_is_playing.set(False)
        ui.update_action_button("mt_btn_play", label="Play")

    # ── Core sampling ────────────────────────────────────────────────────
    def _draw_samples(num_experiments: int):
        m     = _get_m()
        k     = _get_k()
        delta = _get_delta()
        n     = _get_n()
        sigma = _get_sigma()
        alpha = _get_alpha()
        rho   = _get_rho()

        # Ground truth
        is_h1 = np.zeros(m, dtype=bool)
        is_h1[:k] = True
        mu_vec = np.where(is_h1, delta, 0.0)  # (m,)

        # Accumulators (copy from reactive)
        acc_stats = {meth: dict(s) for meth, s in mt_method_stats().items()}
        acc_fwer = dict(mt_fwer_counts())
        acc_fdr = dict(mt_fdr_sums())
        pv_deque = deque(mt_all_pvalues(), maxlen=MAX_DATA)

        last_pvals = None

        for _ in range(num_experiments):
            # ── Generate data ────────────────────────────────────────────
            if rho < 1e-9:
                X = np.random.normal(mu_vec, sigma, size=(n, m))
            else:
                W = np.random.normal(0, 1, (n, 1))
                E = np.random.normal(0, 1, (n, m))
                X = mu_vec + sigma * (np.sqrt(rho) * W + np.sqrt(1 - rho) * E)

            # ── One-sample t-tests (H₀: μ = 0) ─────────────────────────
            means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=1)
            stds = np.maximum(stds, 1e-10)
            t_stats = means / (stds / np.sqrt(n))
            pvalues = 2.0 * stats.t.cdf(-np.abs(t_stats), df=n - 1)
            last_pvals = pvalues

            # ── Apply corrections ────────────────────────────────────────
            corrections = _apply_corrections(pvalues, alpha)

            for meth in _METHODS:
                rejected = corrections[meth]
                fp = int(np.sum(rejected & ~is_h1))
                tp = int(np.sum(rejected & is_h1))
                R = fp + tp

                acc_stats[meth]["fp"] += fp
                acc_stats[meth]["tp"] += tp

                if fp > 0:
                    acc_fwer[meth] += 1

                # FDR: FP/R if R > 0, else 0
                acc_fdr[meth] += (fp / R) if R > 0 else 0.0

            pv_deque.extend(pvalues.tolist())

        # ── Update reactive state ────────────────────────────────────────
        mt_total.set(mt_total() + num_experiments)
        mt_method_stats.set(acc_stats)
        mt_fwer_counts.set(acc_fwer)
        mt_fdr_sums.set(acc_fdr)
        mt_all_pvalues.set(pv_deque)
        mt_last_pvalues.set(last_pvals)
        mt_last_is_h1.set(is_h1)

    # ── Stat text outputs ────────────────────────────────────────────────
    @render.text
    def mt_fwer_val():
        tot = mt_total()
        if tot == 0:
            return "\u2014"
        rate = mt_fwer_counts()["none"] / tot
        return f"{rate:.1%}"

    @render.text
    def mt_fdr_val():
        tot = mt_total()
        if tot == 0:
            return "\u2014"
        fdr = mt_fdr_sums()["bh"] / tot
        return f"{fdr:.1%}"

    @render.text
    def mt_power_val():
        tot = mt_total()
        k = _get_k()
        if tot == 0 or k == 0:
            return "\u2014"
        power = mt_method_stats()["bh"]["tp"] / (tot * k)
        return f"{power:.1%}"

    @render.text
    def mt_total_val():
        return f"{mt_total():,}"

    # ── Chart renderers ──────────────────────────────────────────────────
    @render.ui
    def mt_scatter_plot():
        dark = is_dark()
        try:
            sort_by_rank = input.mt_scatter_x() == "rank"
        except Exception:
            sort_by_rank = True
        try:
            file_drawer = bool(input.mt_file_drawer())
        except Exception:
            file_drawer = False

        fig = draw_mt_pvalue_scatter(
            pvalues=mt_last_pvalues(),
            is_h1=mt_last_is_h1(),
            alpha=_get_alpha(),
            sort_by_rank=sort_by_rank,
            file_drawer=file_drawer,
            dark=dark,
        )
        return _fig_to_ui(fig)

    @render.ui
    def mt_bars_plot():
        fig = draw_mt_correction_bars(
            method_stats=mt_method_stats(),
            total=mt_total(),
            k=_get_k(),
            alpha=_get_alpha(),
            dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def mt_fwer_plot():
        tot = mt_total()
        emp = mt_fwer_counts()["none"] / tot if tot > 0 else None
        fig = draw_mt_fwer_curve(
            m=_get_m(),
            alpha=_get_alpha(),
            empirical_fwer=emp,
            total=tot,
            dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def mt_hist_plot():
        try:
            file_drawer = bool(input.mt_file_drawer())
        except Exception:
            file_drawer = False
        fig = draw_mt_pvalue_hist(
            all_pvalues=list(mt_all_pvalues()),
            alpha=_get_alpha(),
            m=_get_m(),
            file_drawer=file_drawer,
            dark=is_dark(),
        )
        return _fig_to_ui(fig)

    # ── BH step-by-step table ────────────────────────────────────────────
    @render.ui
    def mt_bh_table():
        pvals = mt_last_pvalues()
        is_h1 = mt_last_is_h1()
        if pvals is None:
            dark = is_dark()
            muted = "#64748b" if dark else "#94a3b8"
            return ui.div(
                ui.tags.p("Press Sample to begin",
                          style=f"text-align:center; color:{muted}; "
                                "padding:20px; font-size:0.85rem;"),
            )

        m = len(pvals)
        alpha = _get_alpha()
        dark = is_dark()

        sorted_idx = np.argsort(pvals)
        sorted_p = pvals[sorted_idx]
        sorted_h1 = is_h1[sorted_idx]
        ranks = np.arange(1, m + 1)
        c_m = float(np.sum(1.0 / ranks))

        bh_thresh = ranks * alpha / m
        by_thresh = ranks * alpha / (m * c_m)

        # Find BH cutoff: largest k where p_(k) <= threshold
        below_bh = sorted_p <= bh_thresh
        bh_cutoff = int(np.max(np.where(below_bh)[0])) if below_bh.any() else -1

        # Colors
        bg = "#1e293b" if dark else "#ffffff"
        hd_bg = "#0f172a" if dark else "#f1f5f9"
        txt = "#e2e8f0" if dark else "#1e293b"
        txt2 = "#94a3b8" if dark else "#64748b"
        rej_bg = "rgba(56,189,248,0.12)" if dark else "rgba(56,189,248,0.10)"
        cut_bg = "rgba(249,115,22,0.18)" if dark else "rgba(249,115,22,0.12)"
        border = "#334155" if dark else "#e2e8f0"

        rows = []
        # Show all rows (scrollable)
        for i in range(m):
            rank = i + 1
            p = sorted_p[i]
            bh_t = bh_thresh[i]
            by_t = by_thresh[i]
            test_idx = int(sorted_idx[i]) + 1
            hyp = "H\u2081" if sorted_h1[i] else "H\u2080"
            hyp_col = "#34d399" if sorted_h1[i] else txt2

            if i <= bh_cutoff:
                row_bg = rej_bg
                status = "\u2713"
                status_col = "#38bdf8"
            elif i == bh_cutoff + 1:
                row_bg = cut_bg
                status = "\u2717"
                status_col = "#f97316"
            else:
                row_bg = "transparent"
                status = "\u2717"
                status_col = txt2

            p_col = "#f87171" if (p < alpha and not sorted_h1[i]) else txt

            rows.append(
                f"<tr style='background:{row_bg}'>"
                f"<td style='text-align:center;color:{txt2}'>{rank}</td>"
                f"<td style='color:{hyp_col}'>{hyp} #{test_idx}</td>"
                f"<td style='color:{p_col}'>{p:.4f}</td>"
                f"<td style='color:{txt2}'>{bh_t:.4f}</td>"
                f"<td style='color:{txt2}'>{by_t:.5f}</td>"
                f"<td style='text-align:center;color:{status_col};font-weight:600'>{status}</td>"
                f"</tr>"
            )

        table_html = f"""
        <div class="bh-table-wrap" style="overflow-y:auto; max-height:100%;
             font-family:Inter,sans-serif; font-size:0.72rem;">
          <table style="width:100%; border-collapse:collapse; color:{txt};">
            <thead>
              <tr style="background:{hd_bg}; position:sticky; top:0; z-index:1;">
                <th style="padding:3px 6px; border-bottom:1px solid {border};
                    font-size:0.65rem; color:{txt2}; font-weight:600;">Rank</th>
                <th style="padding:3px 6px; border-bottom:1px solid {border};
                    font-size:0.65rem; color:{txt2}; font-weight:600;">Test</th>
                <th style="padding:3px 6px; border-bottom:1px solid {border};
                    font-size:0.65rem; color:{txt2}; font-weight:600;">p-value</th>
                <th style="padding:3px 6px; border-bottom:1px solid {border};
                    font-size:0.65rem; color:{txt2}; font-weight:600;">BH (i/m\u00b7\u03b1)</th>
                <th style="padding:3px 6px; border-bottom:1px solid {border};
                    font-size:0.65rem; color:{txt2}; font-weight:600;">BY</th>
                <th style="padding:3px 6px; border-bottom:1px solid {border};
                    font-size:0.65rem; color:{txt2}; font-weight:600;">Reject (BH)</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
        """
        return ui.HTML(table_html)
