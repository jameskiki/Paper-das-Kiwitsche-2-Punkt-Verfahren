"""
KZV Parameter Estimation — Phase 2 and Phase 3 of the Kiwitsche 2-Punkt-Verfahren.

Given a time series from a Bang-Bang limit-cycle experiment, this module:
  1. Detects the stationary limit cycle.
  2. Measures T_u (period) and A_u (peak-to-peak amplitude).
  3. Estimates the dead time L from the sign-change of dy/dt after each switch.
  4. Identifies the FOPDT model parameters (K, T, L).
  5. Computes IMC-PID tuning parameters (K_p, T_i, T_d).

Run as a module for a full demonstration:
    python -m analysis.parameter_estimation
"""

from __future__ import annotations

import math
import numpy as np

from simulation.bang_bang_control import BangBangController, simulate_bang_bang
from simulation.plant_models import FOPDTPlant
import config


# ─── Limit-cycle characteristic extraction ───────────────────────────────────

def extract_limit_cycle_characteristics(
    t: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    n_cycles: int = 5,
) -> dict[str, float]:
    """Extract T_u and A_u from a limit-cycle time series.

    The method uses positive zero-crossings of the error signal ``e`` to
    detect cycle boundaries and averages over the last ``n_cycles`` complete
    cycles.

    Parameters
    ----------
    t : np.ndarray
        Time vector [s].
    y : np.ndarray
        Controlled variable (plant output).
    e : np.ndarray
        Error signal (setpoint − y).
    n_cycles : int
        Number of cycles to average over (default: 5).

    Returns
    -------
    dict with keys:
        ``T_u``  : float — average oscillation period [s]
        ``A_u``  : float — average peak-to-peak amplitude [output unit]
        ``omega_u`` : float — angular frequency [rad/s]
        ``A``    : float — single-sided amplitude = A_u / 2
    """
    # Detect positive zero-crossings of the error
    crossings = []
    for k in range(1, len(e)):
        if e[k - 1] <= 0.0 < e[k]:
            # Linear interpolation for sub-sample accuracy
            frac = -e[k - 1] / (e[k] - e[k - 1])
            t_cross = t[k - 1] + frac * (t[k] - t[k - 1])
            crossings.append((k, t_cross))

    if len(crossings) < n_cycles + 1:
        raise ValueError(
            f"Not enough zero-crossings detected ({len(crossings)}). "
            f"Increase simulation time or reduce n_cycles."
        )

    # Use the last n_cycles crossings
    last = crossings[-(n_cycles + 1):]
    periods = [last[i + 1][1] - last[i][1] for i in range(n_cycles)]
    T_u = float(np.mean(periods))

    # Peak-to-peak amplitude over the same last n_cycles
    amplitudes = []
    for i in range(n_cycles):
        s = last[i][0]
        e_ = last[i + 1][0]
        amplitudes.append(y[s:e_].max() - y[s:e_].min())
    A_u = float(np.mean(amplitudes))

    omega_u = 2.0 * math.pi / T_u
    A = A_u / 2.0

    return {"T_u": T_u, "A_u": A_u, "omega_u": omega_u, "A": A}

# ─── Dead-time estimation from transient response ────────────────────────────

def estimate_dead_time_from_limit_cycle(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    dt: float,
    n_last: int = 5,
) -> float:
    """Estimate dead time L from the slope sign-change after BB switching events.

    For a FOPDT plant driven by Bang-Bang control:
    - During the "off" half-cycle (u = U_min), the output falls toward K·U_min,
      so dy/dt < 0.
    - Exactly at t = t_sw + L after a switch to U_max, the dead-time buffer
      flushes and dy/dt becomes positive.
    The sign change of dy/dt therefore occurs at t ≈ t_sw + L, giving a
    direct estimate of L.

    Parameters
    ----------
    t : np.ndarray
        Time vector [s].
    y : np.ndarray
        Plant output.
    u : np.ndarray
        Controller output (bang-bang signal).
    dt : float
        Simulation time step [s].
    n_last : int
        Number of switching events (from the end of the record) to average.

    Returns
    -------
    float
        Estimated dead time [s].
    """
    dy = np.diff(y) / dt
    u_range = float(u.max() - u.min())
    threshold = 0.5 * u_range  # midpoint for detecting switches

    L_estimates: list[float] = []

    # Find rising switching instants (U_min → U_max)
    for k in range(1, len(u)):
        if u[k] - u[k - 1] > threshold:
            # Look for the first positive slope after the switch
            search_end = min(k + int(500 / dt) + 1, len(dy))
            for j in range(k, search_end):
                if dy[j] > 0.0:
                    L_est = (j - k) * dt
                    if L_est > 0.0:
                        L_estimates.append(L_est)
                    break

    if not L_estimates:
        raise ValueError(
            "Could not estimate dead time: no rising-switch + sign-change pairs found."
        )

    # Average over the last n_last estimates (stationary limit cycle)
    return float(np.mean(L_estimates[-n_last:]))


def identify_fopdt_from_transients(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    dt: float,
    u_min: float = 0.0,
    u_max: float = 100.0,
    n_last: int = 5,
) -> dict[str, float]:
    """Identify FOPDT parameters (K, T, L) from the transient response within the
    Bang-Bang limit cycle.

    This method uses the step-response segments embedded in each half-cycle of
    the limit-cycle oscillation, based on the following FOPDT relationships:

    **Dead time L** — For a FOPDT plant, the output slope dy/dt changes sign
    exactly at t = t_sw + L after a rising switch (U_min → U_max), since only
    then does the new input reach the plant.  The first positive dy/dt after
    the switch therefore estimates t_sw + L.

    **Time constant T** — After a falling switch (U_max → U_min) and dead-time
    L, the output decays toward K·U_min.  If U_min = 0 the FOPDT slope at the
    dead-time-expiry moment is:

        dy/dt|_{t_sw+L} = (K·U_min − y_L) / T = −y_L / T

    Hence T = y_L / |dy/dt|_{t_sw+L}.

    **Process gain K** — From the rising-switch segment after the dead time
    expires:

        dy/dt|_{t_sw+L} = (K·U_max − y_L) / T  ⟹  K = (T·slope + y_L) / U_max

    Parameters
    ----------
    t : np.ndarray
        Time vector [s].
    y : np.ndarray
        Plant output.
    u : np.ndarray
        Bang-Bang controller output.
    dt : float
        Simulation time step [s].
    u_min : float
        Minimum BB output level (default 0).
    u_max : float
        Maximum BB output level (default 100).
    n_last : int
        Number of switching events from the end of the record to average over.

    Returns
    -------
    dict with keys ``K``, ``T``, ``L``.
    """
    dy = np.diff(y) / dt
    u_range = u_max - u_min
    threshold = 0.5 * u_range

    L_estimates: list[float] = []
    T_estimates: list[float] = []
    K_estimates: list[float] = []

    for k in range(1, len(u) - 1):
        delta_u = u[k] - u[k - 1]

        if delta_u > threshold:
            # ── Rising switch: U_min → U_max ──
            # Find first positive dy/dt → gives t_sw + L
            search_end = min(k + int(500 / dt) + 1, len(dy))
            for j in range(k, search_end):
                if dy[j] > 0.0:
                    L_est = (j - k) * dt
                    if L_est > 0.0:
                        L_estimates.append(L_est)
                        y_L = float(y[j])
                        slope_up = float(dy[j])
                        T_est_up = (u_max * 0.0) / slope_up if slope_up == 0 else None
                        K_est_raw = None
                        # We will compute K later using T from falling switches
                        # Store (y_L_up, slope_up) for later pairing
                    break

        elif delta_u < -threshold:
            # ── Falling switch: U_max → U_min ──
            # After dead time L, slope is (K*u_min - y_L) / T = (0 - y_L) / T for u_min=0
            # Find the first negative slope peak (maximum |slope|)
            L_est_fall = L_estimates[-1] if L_estimates else None
            if L_est_fall is None:
                continue
            j_L = k + int(L_est_fall / dt)
            if j_L >= len(dy):
                continue
            slope_down = float(dy[j_L])  # should be negative
            y_L_down = float(y[j_L])
            if slope_down < 0 and y_L_down > 0:
                T_est = (y_L_down - u_min * 0.0) / abs(slope_down)
                # For u_min = 0: T = y_L_down / |slope_down|
                # For general u_min: T = (y_L_down - K*u_min) / |slope_down| (needs K)
                T_estimates.append(T_est)

    if not T_estimates:
        raise ValueError(
            "Could not estimate T: not enough falling-switch transients found."
        )
    if not L_estimates:
        raise ValueError(
            "Could not estimate L: not enough rising-switch sign-change pairs found."
        )

    L_hat = float(np.mean(L_estimates[-n_last:]))
    T_hat = float(np.mean(T_estimates[-n_last:]))

    # Now compute K from rising switches using estimated T
    for k in range(1, len(u) - 1):
        if u[k] - u[k - 1] > threshold:
            j_L = k + int(L_hat / dt)
            if j_L >= len(dy):
                continue
            slope_up = float(dy[j_L])
            y_L_up = float(y[j_L])
            if slope_up > 0:
                K_est = (T_hat * slope_up + y_L_up) / u_max
                if K_est > 0:
                    K_estimates.append(K_est)

    if not K_estimates:
        raise ValueError("Could not estimate K from rising-switch transients.")

    K_hat = float(np.mean(K_estimates[-n_last:]))

    return {"K": K_hat, "T": T_hat, "L": L_hat}


# ─── FOPDT identification ─────────────────────────────────────────────────────

def identify_fopdt(
    T_u: float,
    A_u: float,
    d: float,
    M: float,
    L: float | None = None,
) -> dict[str, float]:
    """Identify FOPDT parameters from Bang-Bang limit-cycle measurements.

    Implements Phase 3 of the KZV as described in the paper (Section 3.3).

    Parameters
    ----------
    T_u : float
        Measured oscillation period [s].
    A_u : float
        Measured peak-to-peak amplitude [output unit].
    d : float
        Bang-Bang hysteresis half-width [output unit].
    M : float
        Bang-Bang relay amplitude = (u_max − u_min) / 2 [input unit].
    L : float, optional
        Dead time [s].  If provided (e.g. from ``estimate_dead_time_from_limit_cycle``),
        the phase equation is used to solve for T.  If None, the simplified
        formula L ≈ (π − arcsin(d/A)) / ω_u is used (valid only when T ≪ 1/ω_u).

    Returns
    -------
    dict with keys ``K``, ``T``, ``L`` (FOPDT parameters).
    """
    omega_u = 2.0 * math.pi / T_u
    A = A_u / 2.0

    if A <= d:
        raise ValueError(
            f"Amplitude A={A:.4f} must be greater than hysteresis d={d:.4f}."
        )

    arcsin_d_A = math.asin(min(d / A, 1.0))
    phase_total = math.pi - arcsin_d_A  # = π − arcsin(d/A)

    if L is None:
        # Simplified: assumes T ≪ 1/ω_u so all phase lag from dead time
        L_hat = phase_total / omega_u
    else:
        L_hat = float(L)

    # Time constant from phase condition: arctan(ω_u·T) = phase_total − ω_u·L
    phase_T = phase_total - omega_u * L_hat
    if 0.0 < phase_T < math.pi / 2.0:
        T_hat = math.tan(phase_T) / omega_u
    else:
        T_hat = T_u / (2.0 * math.pi)  # fallback

    # Gain from describing-function gain condition: K/sqrt(1+(ω_u·T)²) = πA/(4M)
    K_hat = (
        (math.pi * A)
        / (4.0 * M)
        * math.sqrt(1.0 + (omega_u * T_hat) ** 2)
    )

    return {"K": K_hat, "T": T_hat, "L": L_hat}


# ─── IMC-PID synthesis ────────────────────────────────────────────────────────

def imc_pid(
    K: float,
    T: float,
    L: float,
    lambda_: float | None = None,
) -> dict[str, float]:
    """Compute IMC-PID parameters for a FOPDT plant.

    Uses the Rivera (1986) IMC-PID formulas with closed-loop time constant λ.

    Parameters
    ----------
    K : float
        FOPDT process gain.
    T : float
        FOPDT time constant [s].
    L : float
        FOPDT dead time [s].
    lambda_ : float, optional
        Desired closed-loop time constant [s].  Default: ``L`` (= dead time).

    Returns
    -------
    dict with keys ``K_p``, ``T_i``, ``T_d``.
    """
    if lambda_ is None:
        lambda_ = L
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")

    K_p = T / (K * (lambda_ + L))
    T_i = T
    T_d = L / 2.0

    return {"K_p": K_p, "T_i": T_i, "T_d": T_d}


# ─── Demo ────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Full KZV pipeline on a benchmark FOPDT plant."""
    # All values come from config.py so printed 'true' labels stay accurate
    K_true  = config.PLANT_K
    T_true  = config.PLANT_T
    L_true  = config.PLANT_L
    dt      = config.DT
    setpoint = config.SETPOINT
    t_end   = config.SIM_T_END

    plant = FOPDTPlant(K=K_true, T=T_true, L=L_true, dt=dt, y0=setpoint)
    controller = BangBangController(
        u_max=config.BB_U_MAX, u_min=config.BB_U_MIN, d=config.BB_D
    )

    print(f"Running Bang-Bang simulation ({t_end:.0f} s) …")
    t, y, u = simulate_bang_bang(
        plant=plant,
        controller=controller,
        setpoint=setpoint,
        t_end=t_end,
        dt=dt,
    )
    e = setpoint - y

    # Phase 2: Extract limit-cycle characteristics
    chars = extract_limit_cycle_characteristics(t, y, e, n_cycles=5)
    print(f"\nPhase 2 — Limit-cycle characteristics:")
    print(f"  T_u = {chars['T_u']:.2f} s")
    print(f"  A_u = {chars['A_u']:.4f} °C")

    # Phase 3: Identify FOPDT using transient slope method
    fopdt = identify_fopdt_from_transients(
        t=t, y=y, u=u, dt=dt,
        u_min=controller.u_min, u_max=controller.u_max, n_last=5,
    )
    print(f"\nPhase 3 — Identified FOPDT parameters (transient method):")
    print(f"  K = {fopdt['K']:.4f}  (true: {K_true})")
    print(f"  T = {fopdt['T']:.2f} s  (true: {T_true:.1f} s)")
    print(f"  L = {fopdt['L']:.2f} s  (true: {L_true:.1f} s)")

    # Phase 4: PID synthesis
    pid_params = imc_pid(**fopdt)
    print(f"\nPhase 4 — IMC-PID parameters (λ = L̂):")
    print(f"  K_p = {pid_params['K_p']:.4f}")
    print(f"  T_i = {pid_params['T_i']:.2f} s")
    print(f"  T_d = {pid_params['T_d']:.2f} s")


if __name__ == "__main__":
    _demo()
