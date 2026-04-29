"""
Central configuration for all paper simulations and figures.

Edit the values here to change parameters across every plot at once.
Run any plotting module from the ``python/`` directory after editing:
    python -m plotting.plot_results
    python -m plotting.plot_intro_visuals
"""

# ── Benchmark (slow) FOPDT plant ─────────────────────────────────────────────
# Representative industrial heating/cooling process.
PLANT_K = 0.4      # process gain           [°C / %]
PLANT_T = 120.0    # time constant          [s]
PLANT_L = 20.0     # dead time              [s]

# ── T-dominated FOPDT plant (low L/T) ───────────────────────────────────────
# L/T ≈ 0.1: lag dominates, smooth near-sinusoidal oscillations.
# Used in intro figures (top row).
LOW_LT_T  = 100.0  # time constant          [s]
LOW_LT_L  = 10.0   # dead time              [s]  → L/T = 0.1

# ── Balanced FOPDT plant (medium L/T) ────────────────────────────────────────
# L/T = 1.0: equal lag and dead time, asymmetric sawtooth oscillations.
# Used in intro figures (middle row).
MID_T   = 30.0     # time constant          [s]
MID_L   = 30.0     # dead time              [s]  → L/T = 1.0

# ── Dead-time-dominant FOPDT plant (high L/T) ────────────────────────────────
# L/T >> 1: oscillation is dominated by the dead time, not the lag.
# Used in intro figures (bottom row).
DEAD_T  = 10.0     # time constant          [s]
DEAD_L  = 100.0     # dead time             [s]  → L/T = 10

# ── Bang-Bang controller ──────────────────────────────────────────────────────
BB_U_MAX = 100.0   # controller output: ON  [%]
BB_U_MIN = 0.0     # controller output: OFF [%]
BB_D     = 0.3     # hysteresis half-band   [°C]

# ── Simulation time step ──────────────────────────────────────────────────────
DT = 0.5           # default step size [s]
# Finer steps used in the phase portrait to keep orbits smooth.
# Rule of thumb: dt << min(T, L) / 10
DT_LOW_LT = 0.1    # for low-L/T plant (L=10s)         [s]
DT_MID  = 0.5      # for medium-L/T plant (L=30s)      [s]
DT_DEAD = 0.1      # for dead-time-dominant plant      [s]

# ── Operating points ──────────────────────────────────────────────────────────
SETPOINT  = 20.0              # nominal setpoint used in results figures  [°C]
SETPOINTS  = [15.0, 20.0, 25.0]  # setpoint sweep used in intro figures   [°C]

# ── Intro duty-cycle figure display ──────────────────────────────────────────
# Each cell auto-scales to 3 complete limit cycles (t_window=None).
# DC_PLOT_WINDOW is kept here in case a fixed window is ever needed.
DC_PLOT_WINDOW = 200.0   # [s]  (unused by default)

# ── Simulation lengths ────────────────────────────────────────────────────────
SIM_T_END = 3000.0   # default bang-bang run length for limit-cycle ID  [s]

# ── Limit-cycle figure ────────────────────────────────────────────────────────
LC_PLOT_WINDOW = 200.0   # trailing window shown in the limit-cycle figure  [s]

# ── Linear step-response figure (Figure 2) ───────────────────────────────────
STEP_T_END    = 1200.0   # total simulation length for the step response    [s]
STEP_PRE_TIME = 100.0    # pre-step settle time                             [s]
STEP_SIZE     = 5.0      # step magnitude                                   [°C]

# ── Setpoint-sweep preview ────────────────────────────────────────────────────
SWEEP_SETPOINTS    = [15.0, 20.0, 25.0, 30.0]   # setpoints for the sweep preview
SWEEP_PLOT_WINDOW  = 700.0   # trailing window shown per subplot            [s]

# ── Nonlinear analysis figures (Figure 3) ────────────────────────────────────
NL_TARGET_SP      = 28.0    # off-nominal operating point for Figure 3b    [°C]
NL_STEP_T_END     = 1200.0  # simulation length for nonlinear step response [s]
NL_STEP_PRE_TIME  = 50.0    # pre-step settle time for nonlinear response   [s]
NL_STEP_SIZE      = 3.0     # step magnitude for nonlinear response         [°C]
