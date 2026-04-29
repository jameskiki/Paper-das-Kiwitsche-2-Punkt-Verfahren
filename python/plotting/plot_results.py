"""
Figure generation for the paper.

Generates and saves all figures referenced in the LaTeX paper to
``paper/figures/``.

Run as a module from the ``python/`` directory:
    python -m plotting.plot_results
"""

from __future__ import annotations

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.bang_bang_control import BangBangController, simulate_bang_bang
from simulation.pid_controller import PIDController
from simulation.plant_models import FOPDTPlant, NonlinearFOPDTPlant
from analysis.parameter_estimation import (
    extract_limit_cycle_characteristics,
    identify_fopdt_from_transients,
    imc_pid,
)
import config

# ─── Configuration ────────────────────────────────────────────────────────────

# Path to the figures directory (relative to this file's location)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(_REPO_ROOT, "paper", "figures")

# Plant parameters
PLANT_K  = config.PLANT_K
PLANT_T  = config.PLANT_T
PLANT_L  = config.PLANT_L
DT       = config.DT
SETPOINT = config.SETPOINT

# Bang-Bang controller parameters
BB_U_MAX = config.BB_U_MAX
BB_U_MIN = config.BB_U_MIN
BB_D     = config.BB_D

# Simulation / plot lengths
SIM_T_END         = config.SIM_T_END
LC_PLOT_WINDOW    = config.LC_PLOT_WINDOW
STEP_T_END        = config.STEP_T_END
STEP_PRE_TIME     = config.STEP_PRE_TIME
STEP_SIZE         = config.STEP_SIZE
SWEEP_SETPOINTS   = config.SWEEP_SETPOINTS
SWEEP_PLOT_WINDOW = config.SWEEP_PLOT_WINDOW
NL_TARGET_SP      = config.NL_TARGET_SP
NL_STEP_T_END     = config.NL_STEP_T_END
NL_STEP_PRE_TIME  = config.NL_STEP_PRE_TIME
NL_STEP_SIZE      = config.NL_STEP_SIZE

# Matplotlib style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 1.5,
})


class NonlinearFOPDTPlant:
    """Synthetic nonlinear FOPDT-like plant for preview figures.

    This keeps the same structure as a delayed first-order plant, but lets
    process gain and time constant vary smoothly with operating point.
    """

    def __init__(
        self,
        K0: float,
        T0: float,
        L: float,
        dt: float = 0.1,
        y0: float = 0.0,
        y_ref: float = 20.0,
        gain_slope: float = -0.015,
        tau_slope: float = 0.02,
    ) -> None:
        self.K0 = K0
        self.T0 = T0
        self.L = L
        self.dt = dt
        self.y = y0
        self.y_ref = y_ref
        self.gain_slope = gain_slope
        self.tau_slope = tau_slope

        self._delay_steps = max(1, int(round(L / dt)))
        self._u_buffer: list[float] = [0.0] * self._delay_steps

    def _K_eff(self) -> float:
        y_norm = (self.y - self.y_ref) / max(abs(self.y_ref), 1.0)
        return max(0.05, self.K0 * (1.0 + self.gain_slope * y_norm * 10.0))

    def _T_eff(self) -> float:
        y_norm = (self.y - self.y_ref) / max(abs(self.y_ref), 1.0)
        return max(5.0, self.T0 * (1.0 + self.tau_slope * y_norm * 10.0))

    def step(self, u: float) -> float:
        u_delayed = self._u_buffer.pop(0)
        self._u_buffer.append(u)

        k_eff = self._K_eff()
        t_eff = self._T_eff()
        dydt = (k_eff * u_delayed - self.y) / t_eff
        self.y += dydt * self.dt
        return self.y


# ─── Helper ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ─── Figure 1: Limit cycle ────────────────────────────────────────────────────

def plot_limit_cycle() -> dict:
    """Plot the Bang-Bang limit cycle and return measured characteristics."""
    plant = FOPDTPlant(K=PLANT_K, T=PLANT_T, L=PLANT_L, dt=DT, y0=SETPOINT)
    ctrl = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
    t, y, u = simulate_bang_bang(plant, ctrl, SETPOINT, t_end=SIM_T_END, dt=DT)
    e = SETPOINT - y

    chars = extract_limit_cycle_characteristics(t, y, e, n_cycles=5)

    # Plot last LC_PLOT_WINDOW s to show stationary cycles clearly
    mask = t >= (t[-1] - LC_PLOT_WINDOW)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(t[mask], y[mask], color="steelblue", label=r"$y(t)$")
    axes[0].axhline(SETPOINT, color="gray", linestyle="--", linewidth=1.0, label=r"Setpoint $r$")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Bang-Bang Limit Cycle — Benchmark Cooling Plant")

    axes[1].step(t[mask], u[mask], color="darkorange", where="post", label=r"$u(t)$")
    axes[1].set_ylabel("Control output (%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-10, 110)
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(25))

    fig.tight_layout()
    _save(fig, "limit_cycle.pdf")
    return chars


# ─── Figure 2: Step response comparison ──────────────────────────────────────

def plot_step_response_comparison(fopdt_params: dict) -> None:
    """Plot closed-loop step responses for KZV-IMC, ZN, and ideal PID."""
    pid_kzv = imc_pid(**fopdt_params)

    # Ziegler-Nichols rules from equivalent ultimate gain K_u and period T_u
    omega_pc = math.pi / fopdt_params["L"]
    K_u_zn = math.sqrt(1.0 + (omega_pc * fopdt_params["T"]) ** 2) / fopdt_params["K"]
    T_u_zn = 2.0 * math.pi / omega_pc
    pid_zn = {
        "K_p": 0.6 * K_u_zn,
        "T_i": 0.5 * T_u_zn,
        "T_d": 0.125 * T_u_zn,
    }

    # "Ideal": IMC with exact plant parameters
    pid_ideal = imc_pid(K=PLANT_K, T=PLANT_T, L=PLANT_L, lambda_=PLANT_L)

    configs = [
        ("KZV-IMC", pid_kzv, "steelblue"),
        ("Ziegler-Nichols", pid_zn, "darkorange"),
        ("Ideal IMC", pid_ideal, "green"),
    ]

    t_end = STEP_T_END
    step_time = STEP_PRE_TIME   # longer pre-step window so PID is fully settled
    step_size = STEP_SIZE
    setpoint_base = SETPOINT
    # Steady-state output needed to hold setpoint (K*u_ss = setpoint)
    u_ss = setpoint_base / PLANT_K

    fig, ax = plt.subplots(figsize=(9, 5))

    metrics: dict[str, dict] = {}

    for label, params, color in configs:
        plant = FOPDTPlant(K=PLANT_K, T=PLANT_T, L=PLANT_L, dt=DT, y0=setpoint_base)
        ctrl = PIDController(
            K_p=params["K_p"],
            T_i=params["T_i"],
            T_d=params["T_d"],
            dt=DT,
            u_min=BB_U_MIN,
            u_max=BB_U_MAX,
            init_output=u_ss,  # bumpless start: integral pre-seeded to hold setpoint
        )
        n_steps = int(t_end / DT)
        t_arr = np.arange(n_steps) * DT
        y_arr = np.empty(n_steps)

        for k in range(n_steps):
            sp = setpoint_base + (step_size if t_arr[k] >= step_time else 0.0)
            u_k = ctrl.compute(sp, plant.y)
            y_arr[k] = plant.step(u_k)

        ax.plot(t_arr - step_time, y_arr - setpoint_base, color=color, label=label)

        # Metrics
        mask_post = t_arr >= step_time
        y_rel = y_arr[mask_post] - setpoint_base
        t_rel = t_arr[mask_post]
        if len(y_rel) > 0:
            overshoot = max(0.0, y_rel.max() - step_size) / step_size * 100.0
            settled = np.where(np.abs(y_rel - step_size) <= 0.05 * step_size)[0]
            t_settle = t_rel[settled[0]] - step_time if len(settled) > 0 else float("nan")
            iae = float(np.sum(np.abs(y_rel - step_size)) * DT)
            metrics[label] = {"Overshoot (%)": overshoot, "Settling time (s)": t_settle, "IAE": iae}

    ax.axhline(step_size, color="gray", linestyle="--", linewidth=1.0, label="Setpoint step")
    ax.axhline(step_size * 1.05, color="lightgray", linestyle=":", linewidth=0.8)
    ax.axhline(step_size * 0.95, color="lightgray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time after step (s)")
    ax.set_ylabel(r"Output change ($^{\circ}$C)")
    ax.set_ylim(-0.5, 9.5)   # clip ZN overshoot — it exceeds this but the key comparison is visible
    ax.set_xlim(-20, t_end - step_time)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "step_response_comparison.pdf")

    # Print metrics table
    print("\nPerformance Metrics:")
    header = f"{'Method':<22} {'Overshoot (%)':>15} {'Settling (s)':>13} {'IAE':>12}"
    print(header)
    print("-" * len(header))
    for lbl, m in metrics.items():
        print(f"{lbl:<22} {m['Overshoot (%)']:>15.1f} {m['Settling time (s)']:>13.1f} {m['IAE']:>12.1f}")


def _run_setpoint_identification(
    setpoint: float,
    nonlinear: bool,
    t_end: float,
) -> dict[str, float] | None:
    if nonlinear:
        plant = NonlinearFOPDTPlant(
            K0=PLANT_K,
            T0=PLANT_T,
            L=PLANT_L,
            dt=DT,
            y0=setpoint,
            y_ref=SETPOINT,
        )
    else:
        plant = FOPDTPlant(K=PLANT_K, T=PLANT_T, L=PLANT_L, dt=DT, y0=setpoint)

    ctrl = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
    t, y, u = simulate_bang_bang(plant, ctrl, setpoint, t_end=t_end, dt=DT)
    e = setpoint - y

    try:
        chars = extract_limit_cycle_characteristics(t, y, e, n_cycles=5)
        fopdt = identify_fopdt_from_transients(
            t=t,
            y=y,
            u=u,
            dt=DT,
            u_min=BB_U_MIN,
            u_max=BB_U_MAX,
            n_last=5,
        )
        fopdt.update({"setpoint": setpoint, "T_u": chars["T_u"], "A_u": chars["A_u"]})
        return fopdt
    except ValueError:
        return None


def plot_setpoint_sweep_preview() -> None:
    """Preview how linear vs nonlinear plants look under setpoint sweeps."""
    setpoints = np.array(SWEEP_SETPOINTS)

    linear_estimates: list[dict[str, float]] = []
    nonlinear_estimates: list[dict[str, float]] = []

    fig_lc, axes_lc = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes_flat = axes_lc.flatten()

    for idx, sp in enumerate(setpoints):
        plant_nl = NonlinearFOPDTPlant(
            K0=PLANT_K,
            T0=PLANT_T,
            L=PLANT_L,
            dt=DT,
            y0=sp,
            y_ref=SETPOINT,
        )
        ctrl_nl = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
        t_nl, y_nl, _ = simulate_bang_bang(plant_nl, ctrl_nl, sp, t_end=SIM_T_END, dt=DT)

        mask = t_nl >= (t_nl[-1] - SWEEP_PLOT_WINDOW)
        ax = axes_flat[idx]
        ax.plot(t_nl[mask], y_nl[mask], color="firebrick", label="Nonlinear")
        ax.axhline(sp, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Setpoint {sp:.0f} °C")
        ax.set_ylabel("y (°C)")
        ax.grid(alpha=0.2)

        lin = _run_setpoint_identification(sp, nonlinear=False, t_end=SIM_T_END)
        nl = _run_setpoint_identification(sp, nonlinear=True, t_end=SIM_T_END)
        if lin is not None:
            linear_estimates.append(lin)
        if nl is not None:
            nonlinear_estimates.append(nl)

    for ax in axes_lc[-1, :]:
        ax.set_xlabel("Time (s)")
    fig_lc.suptitle("Limit-Cycle Preview for a Setpoint Sweep (Nonlinear Plant)")
    fig_lc.tight_layout()
    _save(fig_lc, "setpoint_sweep_limit_cycles_preview.pdf")

    if not linear_estimates or not nonlinear_estimates:
        print("Setpoint sweep preview skipped: not enough valid estimates.")
        return

    sp_lin = np.array([dct["setpoint"] for dct in linear_estimates])
    sp_nl = np.array([dct["setpoint"] for dct in nonlinear_estimates])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    params = ["K", "T", "L"]
    ylabels = ["Estimated K", "Estimated T (s)", "Estimated L (s)"]

    for i, (p, ylabel) in enumerate(zip(params, ylabels)):
        axes[i].plot(
            sp_lin,
            [dct[p] for dct in linear_estimates],
            "o-",
            color="steelblue",
            label="Linear benchmark",
        )
        axes[i].plot(
            sp_nl,
            [dct[p] for dct in nonlinear_estimates],
            "s-",
            color="firebrick",
            label="Synthetic nonlinear",
        )
        axes[i].set_ylabel(ylabel)
        axes[i].grid(alpha=0.25)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Setpoint (°C)")
    fig.suptitle("Identified FOPDT Parameters vs Setpoint")
    fig.tight_layout()
    _save(fig, "setpoint_sweep_identification_preview.pdf")


# ─── Figure 3: Nonlinear plant analysis ──────────────────────────────────────

def plot_nonlinear_analysis() -> None:
    """Generate publication figures for the nonlinear extensions section.

    Produces two figures:
    1. ``nonlinear_param_variation.pdf`` — True K_eff and T_eff vs operating
       point (analytical) overlaid with KZV-identified values from Bang-Bang
       experiments at each setpoint.
    2. ``nonlinear_step_response.pdf`` — Closed-loop step responses comparing:
       (a) a single KZV PID (tuned at the nominal 20 °C setpoint) applied at a
       different operating point, vs (b) a locally re-identified KZV PID tuned
       at the target setpoint.
    """
    # ── Setpoint grid ──────────────────────────────────────────────────────
    setpoints = np.array([15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0])
    ref_plant = NonlinearFOPDTPlant(K0=PLANT_K, T0=PLANT_T, L=PLANT_L,
                                    dt=DT, y_ref=SETPOINT)

    K_true = np.array([ref_plant.K_eff(sp) for sp in setpoints])
    T_true = np.array([ref_plant.T_eff(sp) for sp in setpoints])

    K_est, T_est, L_est, sp_ok = [], [], [], []

    print("  Running KZV identification sweep across setpoints …")
    for sp in setpoints:
        result = _run_setpoint_identification(sp, nonlinear=True, t_end=SIM_T_END)
        if result is not None:
            K_est.append(result["K"])
            T_est.append(result["T"])
            L_est.append(result["L"])
            sp_ok.append(sp)

    sp_ok = np.array(sp_ok)
    K_est = np.array(K_est)
    T_est = np.array(T_est)
    L_est = np.array(L_est)

    # ── Figure 3a: parameter variation ────────────────────────────────────
    y_grid = np.linspace(10.0, 35.0, 200)
    K_curve = np.array([ref_plant.K_eff(y) for y in y_grid])
    T_curve = np.array([ref_plant.T_eff(y) for y in y_grid])

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    axes[0].plot(y_grid, K_curve, color="steelblue", label="True $K_{\\mathrm{eff}}(y)$")
    axes[0].scatter(sp_ok, K_est, color="darkorange", zorder=5, label="KZV estimate $\\hat{K}$")
    axes[0].set_ylabel("Process gain $K$ [°C/%]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(y_grid, T_curve, color="steelblue", label="True $T_{\\mathrm{eff}}(y)$")
    axes[1].scatter(sp_ok, T_est, color="darkorange", zorder=5, label="KZV estimate $\\hat{T}$")
    axes[1].set_ylabel("Time constant $T$ [s]")
    axes[1].set_xlabel("Operating point / setpoint (°C)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Identified FOPDT Parameters vs Operating Point — Nonlinear Plant",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "nonlinear_param_variation.pdf")

    # ── Figure 3b: step responses — nominal vs local KZV ──────────────────
    target_sp = NL_TARGET_SP  # operating point away from nominal

    # Nominal KZV: tuned at setpoint=20 °C, applied at target_sp
    nom_result = _run_setpoint_identification(SETPOINT, nonlinear=True, t_end=SIM_T_END)
    # Local KZV: tuned at target_sp
    loc_result = _run_setpoint_identification(target_sp, nonlinear=True, t_end=SIM_T_END)
    # Ideal: tuned with exact local K/T at target_sp
    ideal_params = {"K": ref_plant.K_eff(target_sp),
                    "T": ref_plant.T_eff(target_sp),
                    "L": PLANT_L}

    if nom_result is None or loc_result is None:
        print("  Skipping nonlinear step response figure (identification failed).")
        return

    def _fopdt_only(d: dict) -> dict:
        return {"K": d["K"], "T": d["T"], "L": d["L"]}

    configs = [
        ("Nominal KZV (at 20 °C)", imc_pid(**_fopdt_only(nom_result)), "steelblue"),
        (f"Local KZV (at {target_sp:.0f} °C)", imc_pid(**_fopdt_only(loc_result)), "darkorange"),
        ("Ideal local IMC", imc_pid(**ideal_params), "green"),
    ]

    t_end_step = NL_STEP_T_END
    step_time = NL_STEP_PRE_TIME
    step_size = NL_STEP_SIZE

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, pid_p, color in configs:
        plant = NonlinearFOPDTPlant(K0=PLANT_K, T0=PLANT_T, L=PLANT_L,
                                    dt=DT, y0=target_sp, y_ref=SETPOINT)
        ctrl = PIDController(K_p=pid_p["K_p"], T_i=pid_p["T_i"], T_d=pid_p["T_d"],
                             dt=DT, u_min=BB_U_MIN, u_max=BB_U_MAX)
        n_steps = int(t_end_step / DT)
        t_arr = np.arange(n_steps) * DT
        y_arr = np.empty(n_steps)
        for k in range(n_steps):
            sp_k = target_sp + (step_size if t_arr[k] >= step_time else 0.0)
            u_k = ctrl.compute(sp_k, plant.y)
            y_arr[k] = plant.step(u_k)

        ax.plot(t_arr - step_time, y_arr - target_sp, color=color, label=label)

    ax.axhline(step_size, color="gray", linestyle="--", linewidth=1.0, label="Setpoint step")
    ax.axhline(step_size * 1.05, color="lightgray", linestyle=":", linewidth=0.8)
    ax.axhline(step_size * 0.95, color="lightgray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time after step (s)")
    ax.set_ylabel("Output change (°C)")
    ax.set_title(f"Nonlinear Plant — Step Response at {target_sp:.0f} °C Operating Point")
    ax.legend()
    ax.set_xlim(-20, t_end_step - step_time)
    fig.tight_layout()
    _save(fig, "nonlinear_step_response.pdf")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== KZV Figure Generation ===\n")

    print("Generating Figure 1: Limit cycle …")
    # Re-run simulation to get full time series for dead-time estimation
    plant = FOPDTPlant(K=PLANT_K, T=PLANT_T, L=PLANT_L, dt=DT, y0=SETPOINT)
    ctrl_ref = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
    t_all, y_all, u_all = simulate_bang_bang(plant, ctrl_ref, SETPOINT, t_end=SIM_T_END, dt=DT)
    e_all = SETPOINT - y_all

    chars = extract_limit_cycle_characteristics(t_all, y_all, e_all, n_cycles=5)
    print(f"  T_u = {chars['T_u']:.2f} s,  A_u = {chars['A_u']:.4f} °C")

    # Save limit-cycle figure with T_u and A_u annotations
    plant2 = FOPDTPlant(K=PLANT_K, T=PLANT_T, L=PLANT_L, dt=DT, y0=SETPOINT)
    ctrl2 = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
    t_lc, y_lc, u_lc = simulate_bang_bang(plant2, ctrl2, SETPOINT, t_end=SIM_T_END, dt=DT)
    mask = t_lc >= (t_lc[-1] - LC_PLOT_WINDOW)

    # Pick two consecutive zero-crossings near the MIDDLE of the displayed window
    e_lc = SETPOINT - y_lc
    xings = []
    for k in range(1, len(e_lc)):
        if e_lc[k - 1] <= 0.0 < e_lc[k]:
            frac = -e_lc[k - 1] / (e_lc[k] - e_lc[k - 1])
            xings.append(t_lc[k - 1] + frac * (t_lc[k] - t_lc[k - 1]))
    t_win_start = t_lc[mask][0]
    t_win_end = t_lc[mask][-1]
    t_center = (t_win_start + t_win_end) / 2.0
    # Filter crossings inside the window and pick the pair closest to the center
    win_xings = [x for x in xings if t_win_start < x < t_win_end]
    mid_idx = min(range(len(win_xings) - 1),
                  key=lambda i: abs((win_xings[i] + win_xings[i + 1]) / 2.0 - t_center))
    tc1, tc2 = win_xings[mid_idx], win_xings[mid_idx + 1]
    y_mid_top = y_lc[mask].max()
    y_mid_bot = y_lc[mask].min()
    t_mid = (tc1 + tc2) / 2.0
    # Place A_u arrow one full period to the left so it doesn't overlap T_u
    t_au = tc1 - (tc2 - tc1) * 0.65

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t_lc[mask], y_lc[mask], color="steelblue", label=r"$y(t)$")
    axes[0].axhline(SETPOINT, color="gray", linestyle="--", linewidth=1.0, label=r"Setpoint $r$")

    # T_u annotation: horizontal double-headed arrow between two crossings
    axes[0].annotate(
        "", xy=(tc2, SETPOINT + 0.4), xytext=(tc1, SETPOINT + 0.4),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    axes[0].text(t_mid, SETPOINT + 0.65, r"$T_u$", ha="center", va="bottom", fontsize=10)

    # A_u annotation: vertical double-headed arrow (separate x position, left of T_u)
    axes[0].annotate(
        "", xy=(t_au, y_mid_bot), xytext=(t_au, y_mid_top),
        arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.2),
    )
    axes[0].text(t_au + 5, (y_mid_top + y_mid_bot) / 2, r"$A_u$",
                 ha="left", va="center", fontsize=10, color="darkred")

    axes[0].set_ylabel(r"Temperature ($^{\circ}$C)")
    axes[0].legend(loc="upper right")

    axes[1].step(t_lc[mask], u_lc[mask], color="darkorange", where="post", label=r"$u(t)$")
    axes[1].set_ylabel("Control output (%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-10, 110)
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(25))
    fig.tight_layout()
    _save(fig, "limit_cycle.pdf")

    print("\nRunning FOPDT identification (transient slope method) …")
    fopdt_params = identify_fopdt_from_transients(
        t=t_all, y=y_all, u=u_all, dt=DT,
        u_min=BB_U_MIN, u_max=BB_U_MAX, n_last=5,
    )
    print(f"  K={fopdt_params['K']:.4f} (true {PLANT_K}), "
          f"T={fopdt_params['T']:.1f} s (true {PLANT_T}), "
          f"L={fopdt_params['L']:.1f} s (true {PLANT_L})")

    print("\nGenerating Figure 2: Step response comparison …")
    plot_step_response_comparison(fopdt_params)

    print("\nGenerating preview: setpoint sweep nonlinearity figures …")
    plot_setpoint_sweep_preview()

    print("\nGenerating Figure 3: Nonlinear plant analysis …")
    plot_nonlinear_analysis()

    print("\nAll figures saved to paper/figures/")


if __name__ == "__main__":
    main()
