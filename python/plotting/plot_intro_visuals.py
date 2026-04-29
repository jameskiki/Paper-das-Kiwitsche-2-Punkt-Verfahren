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

# Low-L/T (T-dominated) plant
FAST_T = config.LOW_LT_T
FAST_L = config.LOW_LT_L

# Medium L/T plant
MID_T = config.MID_T
MID_L = config.MID_L

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


# --- Figure 1: Duty-cycle comparison (3x3 grid) ------------------------------

def _draw_plant_cell(ax, T, L, r, color, t_window: float | None = None):
    """Draw a single y(t) panel combining ON/OFF background and e>0/e<0 shading."""
    t, y, u = _simulate(r, T=T, L=L)
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

    on_mask = um > 50
    y_lo = min(float(ym.min()), r - BB_D * 3)
    y_hi = max(float(ym.max()), r + BB_D * 3)

    # ON/OFF background shading (lightest layer, drawn first)
    ax.fill_between(tm, y_lo, y_hi, where=on_mask,
                    step="post", alpha=0.12, color=color, linewidth=0)

    # Hysteresis band
    ax.fill_between(tm, r - BB_D, r + BB_D,
                    alpha=0.25, color=color, linewidth=0)

    # e>0 and e<0 regions (y outside hysteresis band)
    ax.fill_between(tm, ym, r, where=(ym > r + BB_D),
                    alpha=0.20, color=color, linewidth=0, interpolate=True)
    ax.fill_between(tm, ym, r, where=(ym < r - BB_D),
                    alpha=0.20, color=color, linewidth=0, interpolate=True)

    ax.plot(tm, ym, color=color, linewidth=1.5)
    ax.axhline(r, color="gray", linestyle="--", linewidth=1.0)
    ax.text(0.97, 0.05, f"DC\u202f=\u202f{duty:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=False))


def plot_duty_cycle_setpoints():
    """3-row x 3-col grid: one y(t) panel per plant/setpoint combination.

    ON/OFF periods are shown as a tinted background; the hysteresis band
    and e>0 / e<0 deviation areas are overlaid as filled regions on the
    same axes, eliminating the need for a separate u(t) subplot row.
    """
    plants = [
        (FAST_T, FAST_L,
         f"$L/T={FAST_L/FAST_T:.1f}$\n$T={FAST_T:.0f}$ s, $L={FAST_L:.0f}$ s"),
        (MID_T,  MID_L,
         f"$L/T={MID_L/MID_T:.1f}$\n$T={MID_T:.0f}$ s, $L={MID_L:.0f}$ s"),
        (DEAD_T, DEAD_L,
         f"$L/T={DEAD_L/DEAD_T:.0f}$\n$T={DEAD_T:.0f}$ s, $L={DEAD_L:.0f}$ s"),
    ]

    fig, axes = plt.subplots(
        3, 3, figsize=(10, 8),
        sharey=True,
        sharex="row",
        gridspec_kw={"hspace": 0.38, "wspace": 0.38},
    )

    for col, (r, color) in enumerate(zip(SETPOINTS, COLORS)):
        axes[0, col].set_title(f"$r = {r:.0f}$\N{DEGREE SIGN}C",
                               fontsize=10, color=color, pad=4)

    for row, (T, L, label) in enumerate(plants):
        for col, (r, color) in enumerate(zip(SETPOINTS, COLORS)):
            ax = axes[row, col]
            _draw_plant_cell(ax, T, L, r, color, t_window=None)
            if col == 0:
                ax.set_ylabel(f"$y(t)$ (\N{DEGREE SIGN}C)\n{label}", fontsize=9)
            if row == 2:
                ax.set_xlabel("3 limit cycles", fontsize=9)
            # Never show raw time tick values — scale varies per row
            ax.set_xticklabels([])

    param_str = (
        f"$K={PLANT_K}$  |  "
        f"$u_{{\\max}}={BB_U_MAX:.0f}$, $u_{{\\min}}={BB_U_MIN:.0f}$  |  "
        f"$d={BB_D}$ (hysteresis half-band)"
    )
    fig.text(0.5, -0.01, param_str, ha="center", va="top", fontsize=9,
             color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="lightgray", alpha=0.9))

    fig.suptitle(
        "Duty cycle is set by $K$ and setpoint only \u2014 not by $T$ or $L$",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, "intro_duty_cycle.pdf")


# --- Figure 2: Phase-plane limit-cycle portraits ------------------------------

def plot_phase_portrait():
    """Phase-plane (y, dy/dt) orbits for slow, fast, and dead-time-dominant plants."""
    # Use a finer dt for the fast and dead-time-dominant plants so the orbit
    # has enough points to look smooth (avoid straight-line segments).
    PHASE_DT_LOW  = config.DT_LOW_LT  # L/T=0.1 plant (L=10s)
    PHASE_DT_MID  = config.DT_MID   # L/T=1.0 plant (L=30s)
    PHASE_DT_HIGH = config.DT_DEAD  # L/T=4.0 plant (L=60s)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.8),
        sharey=True,
        gridspec_kw={"wspace": 0.32},
    )

    plant_configs = [
        (FAST_T, FAST_L, PHASE_DT_LOW,
         f"$L/T={FAST_L/FAST_T:.1f}$\n$T={FAST_T:.0f}$ s, $L={FAST_L:.0f}$ s"),
        (MID_T,  MID_L,  PHASE_DT_MID,
         f"$L/T={MID_L/MID_T:.1f}$\n$T={MID_T:.0f}$ s, $L={MID_L:.0f}$ s"),
        (DEAD_T, DEAD_L, PHASE_DT_HIGH,
         f"$L/T={DEAD_L/DEAD_T:.0f}$\n$T={DEAD_T:.0f}$ s, $L={DEAD_L:.0f}$ s  ($L/T={DEAD_L/DEAD_T:.0f}$)"),
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
