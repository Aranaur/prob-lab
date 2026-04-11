# =============================================================================
# Peeking & Sequential Testing Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import brentq
from shiny import reactive, render, ui

from utils import tip
from seq_plots import (
    draw_seq_trajectory,
    draw_seq_boundary,
    draw_seq_stopping_hist,
    draw_seq_error_bars,
)

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


# ── Boundary helpers ─────────────────────────────────────────────────────────
def _obf_boundaries(K: int, alpha: float) -> np.ndarray:
    """O'Brien-Fleming: z_k = z_{α/2} · √(K/k)."""
    z_a2 = stats.norm.ppf(1 - alpha / 2)
    return np.array([z_a2 * np.sqrt(K / k) for k in range(1, K + 1)])


def _hp_boundaries(K: int, alpha: float) -> np.ndarray:
    """Haybittle-Peto: z=3 for interim looks, z_{α/2} for final."""
    z_a2 = stats.norm.ppf(1 - alpha / 2)
    z = np.full(K, 3.0)
    z[-1] = z_a2
    return z


def _pocock_boundary(K: int, alpha: float) -> float:
    """Pocock: constant boundary via Monte Carlo FWER estimate."""
    if K == 1:
        return float(stats.norm.ppf(1 - alpha / 2))

    # Correlation matrix: Corr(Z_i, Z_j) = √(min(i,j)/max(i,j))
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ti, tj = (i + 1) / K, (j + 1) / K
            C[i, j] = np.sqrt(min(ti, tj) / max(ti, tj))

    n_mc = 30_000
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(np.zeros(K), C, size=n_mc)

    def fwer_at(c: float) -> float:
        return float(np.mean(np.any(np.abs(samples) >= c, axis=1)))

    try:
        return float(brentq(lambda c: fwer_at(c) - alpha, 1.5, 6.0, xtol=0.005))
    except ValueError:
        return float(stats.norm.ppf(1 - alpha / (2 * K)))


def _z_to_pval(z: np.ndarray) -> np.ndarray:
    """Two-sided z → p-value."""
    return 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))


# ═════════════════════════════════════════════════════════════════════════════
def seq_server(input, output, session, is_dark):

    MAX_TRAJ = 100

    # ── Reactive state ────────────────────────────────────────────────────
    seq_total       = reactive.value(0)
    seq_peek_rej    = reactive.value(0)
    seq_seq_rej     = reactive.value(0)
    seq_stop_peek: reactive.Value[deque]  = reactive.value(deque(maxlen=10_000))
    seq_stop_seq: reactive.Value[deque]   = reactive.value(deque(maxlen=10_000))
    seq_trajectories: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_TRAJ))
    seq_traj_stops: reactive.Value[deque]   = reactive.value(deque(maxlen=MAX_TRAJ))
    seq_is_playing  = reactive.value(False)
    seq_speed_ms    = reactive.value(0.5)

    # ── Safe helpers ──────────────────────────────────────────────────────
    def _safe(input_id, default):
        try:
            v = getattr(input, input_id)()
            return type(default)(v) if v is not None else default
        except Exception:
            return default

    def _get_N():     return max(int(_safe("seq_N", 200)), 30)
    def _get_n_min(): return max(int(_safe("seq_n_min", 10)), 5)
    def _get_K():     return max(int(_safe("seq_K", 5)), 2)
    def _get_delta(): return _safe("seq_delta", 0.0)
    def _get_sigma(): return max(_safe("seq_sigma", 1.0), 0.1)
    def _get_alpha(): return _safe("seq_alpha", 0.05)

    # ── Cached boundary computation ──────────────────────────────────────
    @reactive.calc
    def _boundaries():
        K = _get_K()
        alpha = _get_alpha()
        try:
            btype = input.seq_boundary()
        except Exception:
            btype = "obf"

        obf_z = _obf_boundaries(K, alpha)
        poc_z = _pocock_boundary(K, alpha)
        hp_z  = _hp_boundaries(K, alpha)

        if btype == "obf":
            active_z = obf_z
        elif btype == "pocock":
            active_z = np.full(K, poc_z)
        else:
            active_z = hp_z

        return {
            "obf_z": obf_z,
            "pocock_z": poc_z,
            "hp_z": hp_z,
            "active_z": active_z,
            "active_pval": _z_to_pval(active_z),
        }

    # ── Dynamic controls (sequential only) ───────────────────────────────
    @render.ui
    def seq_k_control():
        try:
            mode = input.seq_mode()
        except Exception:
            mode = "peeking"
        if mode != "sequential":
            return ui.div()
        return ui.input_numeric(
            "seq_K",
            ui.TagList(
                "Number of looks (K)",
                tip("Equally-spaced interim analyses at n = N·k/K."),
            ),
            value=5, min=2, max=20, step=1, width="100%",
        )

    @render.ui
    def seq_boundary_control():
        try:
            mode = input.seq_mode()
        except Exception:
            mode = "peeking"
        if mode != "sequential":
            return ui.div()
        return ui.input_select(
            "seq_boundary",
            ui.TagList(
                "Boundary type",
                tip(
                    "O\u2019Brien-Fleming: very conservative early, liberal at the final look. "
                    "Pocock: constant boundary at every look. "
                    "Haybittle-Peto: z=3 for interim, z\u2248\u200a1.96 for final."
                ),
            ),
            choices={
                "obf":    "O\u2019Brien-Fleming",
                "pocock": "Pocock",
                "hp":     "Haybittle-Peto",
            },
            selected="obf", width="100%",
        )

    # ── Speed ± ──────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.seq_speed_minus)
    def _sp_dn():
        seq_speed_ms.set(min(seq_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.seq_speed_plus)
    def _sp_up():
        seq_speed_ms.set(max(seq_speed_ms() - 0.05, 0.05))

    # ── Play / Pause ─────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.seq_btn_play)
    def _toggle():
        seq_is_playing.set(not seq_is_playing())
        ui.update_action_button("seq_btn_play",
                                label="Pause" if seq_is_playing() else "Play")

    @reactive.effect
    def _auto():
        if seq_is_playing():
            reactive.invalidate_later(seq_speed_ms())
            with reactive.isolate():
                _draw_samples(1)

    # ── Manual buttons ───────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.seq_btn_sample_1)
    def _s1():
        _draw_samples(1)

    @reactive.effect
    @reactive.event(input.seq_btn_sample_50)
    def _s50():
        _draw_samples(50)

    @reactive.effect
    @reactive.event(input.seq_btn_sample_100)
    def _s100():
        _draw_samples(100)

    # ── Reset ────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.seq_btn_reset, input.seq_mode, input.seq_alpha,
                    input.seq_N, input.seq_n_min)
    def _reset():
        seq_total.set(0)
        seq_peek_rej.set(0)
        seq_seq_rej.set(0)
        seq_stop_peek.set(deque(maxlen=10_000))
        seq_stop_seq.set(deque(maxlen=10_000))
        seq_trajectories.set(deque(maxlen=MAX_TRAJ))
        seq_traj_stops.set(deque(maxlen=MAX_TRAJ))
        seq_is_playing.set(False)
        ui.update_action_button("seq_btn_play", label="Play")

    # ── Core sampling ────────────────────────────────────────────────────
    def _draw_samples(num_exp: int):
        N     = _get_N()
        n_min = _get_n_min()
        delta = _get_delta()
        sigma = _get_sigma()
        alpha = _get_alpha()
        mode  = _safe("seq_mode", "peeking")

        bnd = _boundaries()
        active_z = bnd["active_z"]

        K = _get_K() if mode == "sequential" else 1
        # Look sample sizes
        if mode == "sequential":
            look_ns = np.array([int(N * k / K) for k in range(1, K + 1)])
            look_ns = np.maximum(look_ns, n_min)
        else:
            look_ns = np.arange(n_min, N + 1)

        peek_rej_acc   = 0
        seq_rej_acc    = 0
        stop_peek_acc  = []
        stop_seq_acc   = []
        traj_acc       = deque(seq_trajectories(), maxlen=MAX_TRAJ)
        stop_acc       = deque(seq_traj_stops(), maxlen=MAX_TRAJ)

        for _ in range(num_exp):
            # Generate full sample
            obs = np.random.normal(delta, sigma, N)

            # Running statistics via cumsum (vectorised)
            cum_sum = np.cumsum(obs)
            cum_sq  = np.cumsum(obs ** 2)
            ns_all  = np.arange(1, N + 1, dtype=float)
            means   = cum_sum / ns_all
            # Running variance: Var = (Σx² - n·x̄²) / (n-1)
            variances = (cum_sq - ns_all * means ** 2) / np.maximum(ns_all - 1, 1)
            sds = np.sqrt(np.maximum(variances, 1e-20))
            t_stats = means / (sds / np.sqrt(ns_all))
            # Fix n=1: set t=0 (undefined)
            t_stats[0] = 0.0
            pvals_all = 2.0 * stats.t.cdf(-np.abs(t_stats), df=np.maximum(ns_all - 1, 1))

            # ── Peeking analysis (always, for comparison) ────────────────
            peek_range = slice(max(n_min - 1, 0), N)
            peek_pvals = pvals_all[peek_range]
            peek_ns    = np.arange(n_min, N + 1)

            # First time p < alpha in peeking
            below_peek = peek_pvals < alpha
            if below_peek.any():
                first_idx = int(np.argmax(below_peek))
                peek_rej_acc += 1
                stop_peek_acc.append(int(peek_ns[first_idx]))
            else:
                stop_peek_acc.append(N)

            # ── Sequential analysis (at look_ns) ────────────────────────
            if mode == "sequential":
                look_indices = look_ns - 1  # 0-based
                look_t = t_stats[look_indices]
                look_p = pvals_all[look_indices]
                # Check if |z| exceeds boundary at any look
                exceeded = np.abs(look_t) >= active_z
                if exceeded.any():
                    first_look = int(np.argmax(exceeded))
                    seq_rej_acc += 1
                    stop_seq_acc.append(int(look_ns[first_look]))
                else:
                    stop_seq_acc.append(N)
                # Store trajectory at look points
                traj_acc.append(look_p.copy())
                stop_n = int(look_ns[int(np.argmax(exceeded))]) if exceeded.any() else None
                stop_acc.append(stop_n)
            else:
                # Peeking mode: store continuous trajectory
                traj_acc.append(peek_pvals.copy())
                if below_peek.any():
                    stop_acc.append(int(peek_ns[int(np.argmax(below_peek))]))
                else:
                    stop_acc.append(None)

                # For sequential comparison in peeking mode, use OBF with K=5
                K_ref = 5
                ref_ns = np.array([int(N * k / K_ref) for k in range(1, K_ref + 1)])
                ref_ns = np.maximum(ref_ns, n_min)
                ref_z  = _obf_boundaries(K_ref, alpha)
                ref_idx = ref_ns - 1
                ref_t = t_stats[ref_idx]
                exceeded = np.abs(ref_t) >= ref_z
                if exceeded.any():
                    seq_rej_acc += 1
                    stop_seq_acc.append(int(ref_ns[int(np.argmax(exceeded))]))
                else:
                    stop_seq_acc.append(N)

        # ── Update reactive state ────────────────────────────────────────
        seq_total.set(seq_total() + num_exp)
        seq_peek_rej.set(seq_peek_rej() + peek_rej_acc)
        seq_seq_rej.set(seq_seq_rej() + seq_rej_acc)

        sp = deque(seq_stop_peek(), maxlen=10_000)
        sp.extend(stop_peek_acc)
        seq_stop_peek.set(sp)

        ss = deque(seq_stop_seq(), maxlen=10_000)
        ss.extend(stop_seq_acc)
        seq_stop_seq.set(ss)

        seq_trajectories.set(traj_acc)
        seq_traj_stops.set(stop_acc)

    # ── Stat text outputs ────────────────────────────────────────────────
    @render.text
    def seq_peek_rate():
        t = seq_total()
        if t == 0:
            return "\u2014"
        return f"{seq_peek_rej() / t:.1%}"

    @render.text
    def seq_seq_rate():
        t = seq_total()
        if t == 0:
            return "\u2014"
        return f"{seq_seq_rej() / t:.1%}"

    @render.text
    def seq_avg_stop():
        ss = list(seq_stop_seq())
        if len(ss) == 0:
            return "\u2014"
        return f"{np.mean(ss):.0f}"

    @render.text
    def seq_total_val():
        return f"{seq_total():,}"

    @render.text
    def seq_lbl_peek():
        delta = _safe("seq_delta", 0.0)
        return "PEEKING POWER" if abs(delta) > 1e-9 else "PEEKING TYPE I"

    @render.text
    def seq_lbl_seq():
        delta = _safe("seq_delta", 0.0)
        return "SEQUENTIAL POWER" if abs(delta) > 1e-9 else "SEQUENTIAL FWER"

    @render.text
    def seq_title_error():
        delta = _safe("seq_delta", 0.0)
        return "POWER COMPARISON" if abs(delta) > 1e-9 else "TYPE I ERROR COMPARISON"

    # ── Chart renderers ──────────────────────────────────────────────────
    @render.ui
    def seq_traj_plot():
        dark = is_dark()
        mode = _safe("seq_mode", "peeking")
        N = _get_N()
        n_min = _get_n_min()
        alpha = _get_alpha()
        n_show = int(_safe("seq_n_traj", 20))

        trajs = list(seq_trajectories())
        stops = list(seq_traj_stops())

        # Limit to last n_show
        trajs = trajs[-n_show:] if len(trajs) > n_show else trajs
        stops = stops[-n_show:] if len(stops) > n_show else stops

        if mode == "sequential":
            K = _get_K()
            look_ns = np.array([int(N * k / K) for k in range(1, K + 1)])
            look_ns = np.maximum(look_ns, n_min)
            bnd = _boundaries()
            boundary_pvals = bnd["active_pval"]
        else:
            look_ns = np.arange(n_min, N + 1)
            boundary_pvals = None

        fig = draw_seq_trajectory(
            trajectories=trajs if len(trajs) > 0 else None,
            n_min=n_min,
            N=N,
            alpha=alpha,
            boundary_pvals=boundary_pvals,
            look_ns=look_ns,
            stop_ns=stops,
            dark=dark,
        )
        return _fig_to_ui(fig)

    @render.ui
    def seq_boundary_plot():
        dark = is_dark()
        K = _get_K()
        alpha = _get_alpha()
        bnd = _boundaries()
        fig = draw_seq_boundary(
            K=K,
            alpha=alpha,
            obf_z=bnd["obf_z"],
            pocock_z=bnd["pocock_z"],
            hp_z=bnd["hp_z"],
            dark=dark,
        )
        return _fig_to_ui(fig)

    @render.ui
    def seq_stop_plot():
        dark = is_dark()
        fig = draw_seq_stopping_hist(
            stop_ns_peek=list(seq_stop_peek()),
            stop_ns_seq=list(seq_stop_seq()),
            N=_get_N(),
            dark=dark,
        )
        return _fig_to_ui(fig)

    @render.ui
    def seq_error_plot():
        dark = is_dark()
        t = seq_total()
        alpha = _get_alpha()
        delta = _get_delta()

        peek_fwer = seq_peek_rej() / t if t > 0 else None
        seq_fwer  = seq_seq_rej() / t if t > 0 else None

        # Fixed-N power for comparison (theoretical, only if delta != 0)
        fixed_power = None
        seq_power   = None
        if abs(delta) > 1e-9 and t > 0:
            N = _get_N()
            sigma = _get_sigma()
            se = sigma / np.sqrt(N)
            ncp = delta / se
            df = N - 1
            tc = stats.t.ppf(1 - alpha / 2, df)
            from scipy.stats import nct
            fixed_power = float(nct.cdf(-tc, df, ncp) + 1 - nct.cdf(tc, df, ncp))
            # Sequential power is just empirical seq_fwer (which is power under H1)
            seq_power = seq_fwer

        fig = draw_seq_error_bars(
            alpha=alpha,
            peek_fwer=peek_fwer,
            seq_fwer=seq_fwer if abs(delta) < 1e-9 else None,
            fixed_power=fixed_power,
            seq_power=seq_power,
            dark=dark,
        )
        return _fig_to_ui(fig)
