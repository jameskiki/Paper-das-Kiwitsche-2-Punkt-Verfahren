"""
Figure generation for the paper.

Generates and saves all figures referenced in the LaTeX paper to
``paper/figures/``.

Run as a module from the ``python/`` directory:
    python -m plotting.plot_results
"""

from __future__ import annotations

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from simulation.bang_bang_control import BangBangController, simulate_bang_bang
from simulation.pid_controller import PIDController
from simulation.plant_models import FOPDTPlant
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
# Matplotlib style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "lines.linewidth": 1.5,
})


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

    # Show the last LC_PLOT_WINDOW s to present stationary cycles clearly.
    # Use a relative time axis (t = 0 at the start of the window).
    mask = t >= (t[-1] - LC_PLOT_WINDOW)
    t_w = t[mask] - t[mask][0]
    y_w = y[mask]
    u_w = u[mask]

    # ── Colour palette ────────────────────────────────────────────────────
    C_PROC  = "#2166ac"   # process variable
    C_SP    = "#555555"   # setpoint / band edges
    C_BAND  = "#e8f0f8"   # hysteresis band fill
    C_CTRL  = "#c0392b"   # control output
    C_ANNOT = "#333333"   # annotation ink

    fig, (ax_y, ax_u) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.06},
    )

    # ── Upper panel: process variable ─────────────────────────────────────
    # Hysteresis band
    ax_y.axhspan(SETPOINT - BB_D, SETPOINT + BB_D,
                 color=C_BAND, linewidth=0, zorder=0)
    ax_y.axhline(SETPOINT - BB_D, color=C_SP, linewidth=0.7,
                 linestyle="--", dashes=(4, 4), zorder=1)
    ax_y.axhline(SETPOINT + BB_D, color=C_SP, linewidth=0.7,
                 linestyle="--", dashes=(4, 4), zorder=1)
    # Setpoint
    ax_y.axhline(SETPOINT, color=C_SP, linewidth=1.1,
                 linestyle="--", zorder=2, label=rf"Setpoint $r = {SETPOINT:.0f}$°C")
    # Process variable
    ax_y.plot(t_w, y_w, color=C_PROC, linewidth=1.6, zorder=3, label=r"$y(t)$")

    # ── Annotate period T_u ───────────────────────────────────────────────
    T_u = chars["T_u"]
    A   = chars["A"]
    e_w = SETPOINT - y_w
    pos_cross = np.where((e_w[:-1] <= 0) & (e_w[1:] > 0))[0]

    y_range = y_w.max() - y_w.min()
    # Reserve headroom above the signal for the T_u arrow + label
    ax_y.set_ylim(y_w.min() - 0.06 * y_range, y_w.max() + 0.38 * y_range)

    if len(pos_cross) >= 2:
        t1 = float(np.interp(0, [e_w[pos_cross[-2]], e_w[pos_cross[-2] + 1]],
                             [t_w[pos_cross[-2]], t_w[pos_cross[-2] + 1]]))
        t2 = float(np.interp(0, [e_w[pos_cross[-1]], e_w[pos_cross[-1] + 1]],
                             [t_w[pos_cross[-1]], t_w[pos_cross[-1] + 1]]))
        arrow_y = y_w.max() + 0.16 * y_range
        ax_y.annotate(
            "", xy=(t2, arrow_y), xytext=(t1, arrow_y),
            arrowprops=dict(arrowstyle="<->", color=C_ANNOT,
                            lw=1.2, shrinkA=0, shrinkB=0),
        )
        ax_y.text(
            (t1 + t2) / 2, arrow_y + 0.04 * y_range,
            rf"$T_u = {T_u:.1f}$ s",
            ha="center", va="bottom", fontsize=8.5, color=C_ANNOT,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    # ── Annotate peak-to-peak amplitude A_u ──────────────────────────────
    # Place the arrow at ~65 % of the window where the signal is centred;
    # text sits to the right inside the axes with a white background box.
    # shrinkA/B=0 so arrowheads land exactly on the peak/trough data values.
    x_ann = t_w[-1] * 0.65
    ax_y.annotate(
        "", xy=(x_ann, SETPOINT + A), xytext=(x_ann, SETPOINT - A),
        arrowprops=dict(arrowstyle="<->", color=C_ANNOT,
                        lw=1.2, shrinkA=0, shrinkB=0),
    )
    ax_y.text(
        x_ann + 0.018 * t_w[-1], SETPOINT,
        rf"$A_u = {2*A:.2f}$ °C",
        ha="left", va="center", fontsize=8.5, color=C_ANNOT,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )

    ax_y.set_ylabel("Temperature (°C)")
    ax_y.legend(loc="upper left", framealpha=0.9, fontsize=8.5,
                handlelength=1.6, borderpad=0.6)
    ax_y.set_title("Bang-Bang Limit Cycle — Benchmark Cooling Plant", pad=6)
    ax_y.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax_y.grid(True, which="major", linewidth=0.5, color="#dddddd", zorder=0)
    ax_y.grid(True, which="minor", linewidth=0.25, color="#eeeeee", zorder=0)

    # ── Lower panel: control output ───────────────────────────────────────
    ax_u.fill_between(t_w, u_w, step="post",
                      color=C_CTRL, alpha=0.15, linewidth=0, zorder=1)
    ax_u.step(t_w, u_w, color=C_CTRL, where="post",
              linewidth=1.5, zorder=2, label=r"$u(t)$")
    # ON / OFF text labels — both at the same vertical centre (y = 50 %)
    # Pick the horizontal centre of the longest continuous run in each state.
    threshold = (BB_U_MAX + BB_U_MIN) / 2
    for state_val, label_str in [(BB_U_MAX, "ON"), (BB_U_MIN, "OFF")]:
        state_mask = u_w == state_val
        if not state_mask.any():
            continue
        # Find all contiguous runs in this state
        padded = np.concatenate(([False], state_mask, [False]))
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends   = np.where( padded[:-1] & ~padded[1:])[0]
        # Pick the longest run
        lengths = ends - starts
        best = np.argmax(lengths)
        t_center = (t_w[starts[best]] + t_w[min(ends[best], len(t_w) - 1)]) / 2
        ax_u.text(
            t_center, 50, label_str,
            ha="center", va="center", fontsize=8.0,
            color=C_CTRL, alpha=0.65,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.6),
        )

    ax_u.set_ylabel("Control output (%)")
    ax_u.set_xlabel("Time (s)")
    ax_u.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax_u.set_ylim(-12, 115)
    ax_u.set_yticks([0, 50, 100])
    ax_u.yaxis.set_minor_locator(ticker.NullLocator())
    ax_u.grid(True, which="major", linewidth=0.5, color="#dddddd", zorder=0)

    # ── Vertical markers at switching events ──────────────────────────────
    # Use a long-dash pattern (8 pt on, 4 pt off) in a medium gray so the
    # markers are clearly intentional but don't compete with the data lines.
    sw_idx = np.where(np.diff(u_w) != 0)[0]
    for i in sw_idx:
        ts = t_w[i]
        ax_y.axvline(ts, color="#999999", linewidth=0.8,
                     linestyle=(0, (6, 4)), zorder=1)
        ax_u.axvline(ts, color="#999999", linewidth=0.8,
                     linestyle=(0, (6, 4)), zorder=1)

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
    fig.suptitle(
        "KZV-IMC vs Ziegler-Nichols vs Ideal IMC \u2014 Closed-Loop Step Response",
        fontsize=11, y=1.005,
    )
    param_str = (
        f"$K={PLANT_K}$, $T={PLANT_T:.0f}$ s, $L={PLANT_L:.0f}$ s  |  "
        f"$u_{{\\max}}={BB_U_MAX:.0f}$, $u_{{\\min}}={BB_U_MIN:.0f}$  |  "
        f"step = +{STEP_SIZE:.0f}\N{DEGREE SIGN}C at $t=0$  |  shaded band = \u00b15\u202f%%"
    )
    fig.text(0.5, -0.01, param_str, ha="center", va="top", fontsize=9,
             color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="lightgray", alpha=0.9))
    fig.tight_layout(rect=[0, 0, 1, 1])
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
    t_end: float,
    K: float = PLANT_K,
    T: float = PLANT_T,
    L: float = PLANT_L,
    dt: float = DT,
) -> dict[str, float] | None:
    plant = FOPDTPlant(K=K, T=T, L=L, dt=dt, y0=setpoint)
    ctrl = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
    t, y, u = simulate_bang_bang(plant, ctrl, setpoint, t_end=t_end, dt=dt)
    e = setpoint - y

    try:
        chars = extract_limit_cycle_characteristics(t, y, e, n_cycles=5)
        fopdt = identify_fopdt_from_transients(
            t=t,
            y=y,
            u=u,
            dt=dt,
            u_min=BB_U_MIN,
            u_max=BB_U_MAX,
            n_last=5,
        )
        fopdt.update({"setpoint": setpoint, "T_u": chars["T_u"], "A_u": chars["A_u"]})
        return fopdt
    except ValueError:
        return None


# ─── Figure 3: PID setpoint comparison (mirrors intro duty-cycle grid) ───────

_PID_COLORS = ["#1565C0", "#2E7D32", "#BF360C"]  # same palette as intro figures


def _draw_pid_cell(
    ax: plt.Axes,
    setpoint: float,
    color: str,
    fopdt_params: dict,
    plant_K: float = PLANT_K,
    plant_T: float = PLANT_T,
    plant_L: float = PLANT_L,
) -> None:
    """Draw one PID step-response panel, mirroring the style of _draw_plant_cell."""
    step_size = STEP_SIZE
    step_time = STEP_PRE_TIME
    t_end = STEP_T_END

    # Steady-state control output to hold the pre-step operating point
    u_ss = max(BB_U_MIN, min(BB_U_MAX, (setpoint - step_size) / plant_K))

    # Three controllers: KZV-IMC, Ziegler-Nichols, Ideal IMC
    pid_kzv = imc_pid(
        K=fopdt_params["K"], T=fopdt_params["T"], L=fopdt_params["L"],
        lambda_=fopdt_params["L"],
    )

    omega_pc = math.pi / max(fopdt_params["L"], 1e-6)
    K_u_zn = math.sqrt(1.0 + (omega_pc * fopdt_params["T"]) ** 2) / max(fopdt_params["K"], 1e-6)
    T_u_zn = 2.0 * math.pi / omega_pc
    pid_zn = {"K_p": 0.6 * K_u_zn, "T_i": 0.5 * T_u_zn, "T_d": 0.125 * T_u_zn}

    pid_ideal = imc_pid(K=plant_K, T=plant_T, L=plant_L, lambda_=plant_L)

    configs = [
        ("KZV-IMC",         pid_kzv,   color,    "-",   2.0),
        ("Ziegler-Nichols", pid_zn,    color,    "--",  1.5),
        ("Ideal IMC",       pid_ideal, color,    ":",   1.5),
    ]

    # ±5 % settling band shading (mirrors hysteresis band in duty-cycle plot)
    band = 0.05 * step_size
    ax.fill_between(
        [-step_time, t_end - step_time],
        setpoint - band, setpoint + band,
        alpha=0.20, color=color, linewidth=0,
    )
    ax.axhline(setpoint, color="gray", linestyle="--", linewidth=1.0)

    n_steps = int(t_end / DT)
    t_arr = np.arange(n_steps) * DT

    for label, params, lcolor, ls, lw in configs:
        plant = FOPDTPlant(K=plant_K, T=plant_T, L=plant_L, dt=DT,
                           y0=setpoint - step_size)
        ctrl = PIDController(
            K_p=params["K_p"], T_i=params["T_i"], T_d=params["T_d"],
            dt=DT, u_min=BB_U_MIN, u_max=BB_U_MAX,
            init_output=u_ss,
        )
        y_arr = np.empty(n_steps)
        for k in range(n_steps):
            sp_k = setpoint if t_arr[k] >= step_time else setpoint - step_size
            y_arr[k] = plant.step(ctrl.compute(sp_k, plant.y))

        ax.plot(t_arr - step_time, y_arr,
                color=lcolor, linestyle=ls, linewidth=lw, label=label, alpha=0.85)

    y_lo = setpoint - step_size - 0.5
    y_hi = setpoint + step_size * 0.6
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=False))


def plot_pid_setpoint_comparison() -> None:
    """3×3 grid of PID step responses mirroring ``intro_duty_cycle.pdf``.

    Rows  — three plants: T-dominated (L/T=0.1), balanced (L/T=1), dead-time
            dominant (L/T=10), matching the intro figure row order.
    Columns — three setpoints (15, 20, 25 °C) in the same colors as the intro.

    Each cell: KZV identification run on that plant/setpoint, then three
    closed-loop step responses overlaid — KZV-IMC (solid), Ziegler-Nichols
    (dashed), Ideal IMC (dotted).
    """
    setpoints = config.SETPOINTS
    colors = _PID_COLORS

    plants = [
        (config.LOW_LT_T,  config.LOW_LT_L,
         f"$L/T={config.LOW_LT_L/config.LOW_LT_T:.1f}$\n"
         f"$T={config.LOW_LT_T:.0f}$ s, $L={config.LOW_LT_L:.0f}$ s"),
        (config.MID_T, config.MID_L,
         f"$L/T={config.MID_L/config.MID_T:.1f}$\n"
         f"$T={config.MID_T:.0f}$ s, $L={config.MID_L:.0f}$ s"),
        (config.DEAD_T, config.DEAD_L,
         f"$L/T={config.DEAD_L/config.DEAD_T:.0f}$\n"
         f"$T={config.DEAD_T:.0f}$ s, $L={config.DEAD_L:.0f}$ s"),
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(10, 8),
        sharey=False,
        sharex="row",
        gridspec_kw={"hspace": 0.38, "wspace": 0.38},
    )

    for col, (r, color) in enumerate(zip(setpoints, colors)):
        axes[0, col].set_title(f"$r = {r:.0f}$\N{DEGREE SIGN}C",
                               fontsize=10, color=color, pad=4)

    for row, (pT, pL, label) in enumerate(plants):
        for col, (r, color) in enumerate(zip(setpoints, colors)):
            ax = axes[row, col]

            print(f"  KZV ID: plant L/T={pL/pT:.1f}, setpoint {r:.0f} °C …")
            result = _run_setpoint_identification(
                r, t_end=SIM_T_END,
                K=PLANT_K, T=pT, L=pL,
            )
            if result is None:
                ax.text(0.5, 0.5, "ID failed", transform=ax.transAxes,
                        ha="center", va="center", color="red")
                continue

            _draw_pid_cell(ax, r, color, result,
                           plant_K=PLANT_K, plant_T=pT, plant_L=pL)

            if col == 0:
                ax.set_ylabel(f"$y(t)$ (\N{DEGREE SIGN}C)\n{label}", fontsize=9)
                if row == 0:
                    ax.legend(loc="lower right", fontsize=8)
            if row == 2:
                ax.set_xlabel(f"+{STEP_SIZE:.0f}\N{DEGREE SIGN}C step", fontsize=9)
            ax.set_xticklabels([])

    param_str = (
        f"$K={PLANT_K}$  |  "
        f"$u_{{\\max}}={BB_U_MAX:.0f}$, $u_{{\\min}}={BB_U_MIN:.0f}$  |  "
        f"shaded band = ±5\u202f% of step  |  "
        f"solid = KZV-IMC, dashed = ZN, dotted = Ideal IMC"
    )
    fig.text(0.5, -0.01, param_str, ha="center", va="top", fontsize=9,
             color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="lightgray", alpha=0.9))

    fig.suptitle(
        "KZV-IMC vs Ziegler-Nichols vs Ideal IMC — closed-loop step response",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, "results_pid_comparison.pdf")


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

    # L annotation: dead time shown on the y(t) panel.
    # At a rising edge of u (OFF→ON), y does not respond until L seconds later.
    # Show two vertical dashed lines and a spanning arrow.
    t_win = t_lc[mask]
    u_win = u_lc[mask]
    y_win = y_lc[mask]
    rising_edges = [i for i in range(1, len(u_win))
                    if u_win[i - 1] < 50 and u_win[i] >= 50]
    # Use the second rising edge (if available) so there's room to the left
    _re_idx = rising_edges[1] if len(rising_edges) >= 2 else rising_edges[0] if rising_edges else None
    if _re_idx is not None:
        t_edge = t_win[_re_idx]
        t_resp = t_edge + PLANT_L
        y_lo_ann = float(y_win.min())
        y_ann = y_lo_ann - 0.3 * (float(y_win.max()) - float(y_win.min()))
        axes[0].axvline(t_edge, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
        axes[0].axvline(t_resp, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
        axes[0].annotate(
            "", xy=(t_resp, y_lo_ann), xytext=(t_edge, y_lo_ann),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
            annotation_clip=False,
        )
        axes[0].text((t_edge + t_resp) / 2, y_lo_ann - 0.15, r"$L$",
                     ha="center", va="top", fontsize=10)

    axes[0].set_ylabel(r"Temperature ($^{\circ}$C)")
    axes[0].legend(loc="upper right")

    axes[1].step(t_lc[mask], u_lc[mask], color="darkorange", where="post", label=r"$u(t)$")
    axes[1].set_ylabel("Control output (%)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(-10, 110)
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(25))
    # Mirror the vertical reference lines on u(t) panel for visual continuity
    if _re_idx is not None:
        axes[1].axvline(t_edge, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
        axes[1].axvline(t_resp, color="black", linestyle=":", linewidth=0.9, alpha=0.7)

    fig.suptitle(
        "Bang-Bang Limit Cycle \u2014 Benchmark Cooling Plant",
        fontsize=11, y=1.005,
    )
    param_str = (
        f"$K={PLANT_K}$, $T={PLANT_T:.0f}$ s, $L={PLANT_L:.0f}$ s  |  "
        f"$u_{{\\max}}={BB_U_MAX:.0f}$, $u_{{\\min}}={BB_U_MIN:.0f}$  |  "
        f"$d={BB_D}$ (hysteresis half-band)"
    )
    fig.text(0.5, -0.01, param_str, ha="center", va="top", fontsize=9,
             color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="lightgray", alpha=0.9))
    fig.tight_layout(rect=[0, 0, 1, 1])
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

    print("\nGenerating Figure 3: PID setpoint comparison …")
    plot_pid_setpoint_comparison()

    print("\nAll figures saved to paper/figures/")


if __name__ == "__main__":
    main()
