"""
Introduction figures for the paper.

Figure 1 - intro_duty_cycle.pdf
    Three-plant, three-setpoint grid showing how the duty cycle (ON-time fraction)
    is determined by K and setpoint only, while T and L control the oscillation
    speed and amplitude. Plants: slow (T=120s, L=20s), fast (T=30s, L=5s), and
    dead-time dominant (T=15s, L=60s, L/T=4). Shaded hysteresis band on y(t)
    panels; shaded ON-periods on u(t) panels.

Figure 2 - intro_phase_portrait.pdf
    Phase-plane (y vs. dy/dt) limit-cycle orbits at the same three setpoints
    for all three plants, with the enclosed area of each orbit shaded to
    highlight the distinct "fingerprints" that encode plant information.

Run as a module from the ``python/`` directory:
    python -m plotting.plot_intro_visuals
"""

from __future__ import annotations

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.bang_bang_control import BangBangController, simulate_bang_bang
from simulation.plant_models import FOPDTPlant
import config

# --- Configuration ------------------------------------------------------------

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
FIGURES_DIR = os.path.join(_REPO_ROOT, "paper", "figures")

# Benchmark (slow) plant
PLANT_K = config.PLANT_K
PLANT_T = config.PLANT_T
PLANT_L = config.PLANT_L
DT      = config.DT

# Fast plant
FAST_T = config.FAST_T
FAST_L = config.FAST_L

# Dead-time-dominant plant
DEAD_T = config.DEAD_T
DEAD_L = config.DEAD_L

BB_U_MAX  = config.BB_U_MAX
BB_U_MIN  = config.BB_U_MIN
BB_D      = config.BB_D

SETPOINTS  = config.SETPOINTS
SIM_T_END  = config.SIM_T_END
DC_PLOT_WINDOW = config.DC_PLOT_WINDOW
COLORS = ["#1565C0", "#2E7D32", "#BF360C"]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "lines.linewidth": 1.5,
})

# --- Helpers ------------------------------------------------------------------

def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def _simulate(r, T=None, L=None, dt=None):
    if T is None:
        T = PLANT_T
    if L is None:
        L = PLANT_L
    if dt is None:
        dt = DT
    t_end = max(SIM_T_END, 25 * T)
    plant = FOPDTPlant(K=PLANT_K, T=T, L=L, dt=dt, y0=r)
    ctrl = BangBangController(u_max=BB_U_MAX, u_min=BB_U_MIN, d=BB_D)
    return simulate_bang_bang(plant, ctrl, r, t_end=t_end, dt=dt)


def _last_n_cycles(t, y, r, n=3):
    e = r - y
    crossings = [i for i in range(1, len(e)) if e[i - 1] < 0 and e[i] >= 0]
    if len(crossings) < n + 1:
        return 0, len(t) - 1
    return crossings[-(n + 1)], crossings[-1]


# --- Figure 1: Duty-cycle comparison (two plants) ----------------------------

def _draw_plant_row(axes_y, axes_u, T, L, row_label, t_window: float | None = None):
    """Populate one pair of y(t)/u(t) axis rows for a given plant.

    Parameters
    ----------
    t_window : float or None
        If given, show the last *t_window* seconds instead of the last
        3 complete cycles.  Pass the same value to both the slow and fast
        plant rows so the x-axis spans are identical and the difference in
        oscillation speed is immediately visible.
    """
    for col, (r, color) in enumerate(zip(SETPOINTS, COLORS)):
        t, y, u = _simulate(r, T=T, L=L)

        # Duty cycle computed analytically: DC = r / (K * U_max).
        # At steady state the average plant output must equal r, so
        # DC * U_max * K = r regardless of T, L, or hysteresis band.
        # This is exactly the invariance the figure is meant to illustrate.
        duty = (r / (PLANT_K * BB_U_MAX)) * 100.0

        if t_window is not None:
            mask = t >= (t[-1] - t_window)
            tm = t[mask] - t[mask][0]
            ym = y[mask]
            um = u[mask]
        else:
            i0, i1 = _last_n_cycles(t, y, r, n=3)
            tm = t[i0:i1 + 1] - t[i0]
            ym = y[i0:i1 + 1]
            um = u[i0:i1 + 1]

        ax_y = axes_y[col]
        # Hysteresis band
        ax_y.fill_between(tm, r - BB_D, r + BB_D,
                          alpha=0.20, color=color, linewidth=0)
        # Tracking error region
        ax_y.fill_between(tm, ym, r, where=(ym > r + BB_D),
                          alpha=0.10, color=color, linewidth=0, interpolate=True)
        ax_y.fill_between(tm, ym, r, where=(ym < r - BB_D),
                          alpha=0.10, color=color, linewidth=0, interpolate=True)
        ax_y.plot(tm, ym, color=color, linewidth=1.5)
        ax_y.axhline(r, color="gray", linestyle="--", linewidth=1.0)
        ax_y.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=False))
        if col == 0:
            ax_y.set_ylabel(f"$y(t)$ (\N{DEGREE SIGN}C)\n{row_label}", fontsize=9)

        ax_u = axes_u[col]
        ax_u.fill_between(tm, BB_U_MIN, um, where=(um > 50),
                          step="post", alpha=0.30, color=color, linewidth=0)
        ax_u.step(tm, um, color=color, where="post", linewidth=1.5)
        ax_u.text(0.97, 0.82, f"DC = {duty:.0f}%",
                  transform=ax_u.transAxes, ha="right", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="gray", alpha=0.85))
        if col == 0:
            ax_u.set_ylabel("$u(t)$ (%)", fontsize=9)
        ax_u.set_ylim(-12, 112)
        ax_u.yaxis.set_major_locator(ticker.MultipleLocator(50))


def plot_duty_cycle_setpoints():
    """6-row x 3-col grid: fast / slow / dead-time-dominant plant rows."""
    fig, axes = plt.subplots(
        6, 3, figsize=(10, 11.5),
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.6, 1.0, 1.6, 1.0],
                     "hspace": 0.15, "wspace": 0.35},
    )

    for col, r in enumerate(SETPOINTS):
        axes[0, col].set_title(f"$r = {r:.0f}$\N{DEGREE SIGN}C",
                               fontsize=10, color=COLORS[col], pad=4)

    # Shared x-axis window for all three plant rows (set DC_PLOT_WINDOW in config.py)
    T_WINDOW = DC_PLOT_WINDOW

    _draw_plant_row(
        axes_y=[axes[0, c] for c in range(3)],
        axes_u=[axes[1, c] for c in range(3)],
        T=FAST_T, L=FAST_L,
        row_label=f"Fast\n$T={FAST_T:.0f}$ s, $L={FAST_L:.0f}$ s",
        t_window=T_WINDOW,
    )
    for c in range(3):
        axes[1, c].set_xlabel("Time (s)", fontsize=9)
        axes[0, c].set_xticklabels([])

    fig.text(
        0.5, 0.664,
        (f"Slow plant:  $T = {PLANT_T:.0f}$ s,  $L = {PLANT_L:.0f}$ s"
         f"  \u2014  same $K = {PLANT_K}$  \u2192  same duty cycles"),
        ha="center", va="center", fontsize=9, style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#aaaaaa"),
    )

    _draw_plant_row(
        axes_y=[axes[2, c] for c in range(3)],
        axes_u=[axes[3, c] for c in range(3)],
        T=PLANT_T, L=PLANT_L,
        row_label=f"Slow\n$T={PLANT_T:.0f}$ s, $L={PLANT_L:.0f}$ s",
        t_window=T_WINDOW,
    )
    for c in range(3):
        axes[3, c].set_xlabel("Time (s)", fontsize=9)
        axes[2, c].set_xticklabels([])

    fig.text(
        0.5, 0.330,
        (f"Dead-time-dominant:  $T = {DEAD_T:.0f}$ s,  $L = {DEAD_L:.0f}$ s"
         f"  ($L/T = {DEAD_L/DEAD_T:.0f}$)  \u2014  same duty cycles"),
        ha="center", va="center", fontsize=9, style="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#aaaaaa"),
    )

    _draw_plant_row(
        axes_y=[axes[4, c] for c in range(3)],
        axes_u=[axes[5, c] for c in range(3)],
        T=DEAD_T, L=DEAD_L,
        row_label=f"Dead-time\ndominant\n$T={DEAD_T:.0f}$ s, $L={DEAD_L:.0f}$ s",
        t_window=T_WINDOW,
    )
    for c in range(3):
        axes[5, c].set_xlabel("Time (s)", fontsize=9)
        axes[4, c].set_xticklabels([])

    fig.suptitle(
        "Duty cycle is set by $K$ and setpoint only \u2014 not by $T$ or $L$",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(hspace=0.18)
    _save(fig, "intro_duty_cycle.pdf")


# --- Figure 2: Phase-plane limit-cycle portraits ------------------------------

def plot_phase_portrait():
    """Phase-plane (y, dy/dt) orbits for slow, fast, and dead-time-dominant plants."""
    # Use a finer dt for the fast and dead-time-dominant plants so the orbit
    # has enough points to look smooth (avoid straight-line segments).
    PHASE_DT_SLOW = config.DT       # slow plant — default is fine
    PHASE_DT_FAST = config.DT_FAST  # fast plant: L=5s needs many more steps per delay
    PHASE_DT_DEAD = config.DT_DEAD  # dead-time-dominant: shorter T needs finer resolution

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.8),
        sharey=True,
        gridspec_kw={"wspace": 0.32},
    )

    plant_configs = [
        (PLANT_T, PLANT_L, PHASE_DT_SLOW,
         f"Slow plant\n$T={PLANT_T:.0f}$ s, $L={PLANT_L:.0f}$ s"),
        (FAST_T,  FAST_L,  PHASE_DT_FAST,
         f"Fast plant\n$T={FAST_T:.0f}$ s, $L={FAST_L:.0f}$ s"),
        (DEAD_T,  DEAD_L,  PHASE_DT_DEAD,
         f"Dead-time dominant\n$T={DEAD_T:.0f}$ s, $L={DEAD_L:.0f}$ s  ($L/T={DEAD_L/DEAD_T:.0f}$)"),
    ]

    for ax, (T, L, phase_dt, title) in zip(axes, plant_configs):
        for r, color in zip(SETPOINTS, COLORS):
            t, y, u = _simulate(r, T=T, L=L, dt=phase_dt)
            i0, i1 = _last_n_cycles(t, y, r, n=2)
            ym = y[i0:i1 + 1]
            dym = np.gradient(ym, phase_dt)

            ax.fill(ym, dym, alpha=0.15, color=color, linewidth=0)
            ax.plot(ym, dym, color=color, linewidth=1.5,
                    label=f"$r = {r:.0f}$\N{DEGREE SIGN}C")
            ax.axvline(r, color=color, linestyle=":", linewidth=0.9, alpha=0.55)

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_xlabel("$y(t)$ (\N{DEGREE SIGN}C)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\dot{y}(t)$ (\N{DEGREE SIGN}C s$^{-1}$)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Phase-plane limit-cycle orbits (shaded = orbit interior, dotted = setpoint)",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    _save(fig, "intro_phase_portrait.pdf")


# --- Entry point --------------------------------------------------------------

def main():
    plot_duty_cycle_setpoints()
    plot_phase_portrait()


if __name__ == "__main__":
    main()
