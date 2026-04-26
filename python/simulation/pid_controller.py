"""
PID controller implementation for KZV closed-loop simulations.
"""

from __future__ import annotations


class PIDController:
    """Parallel-form PID controller with anti-windup.

    Transfer function (continuous):
        C(s) = K_p * (1 + 1/(T_i*s) + T_d*s)

    Discrete-time implementation uses backward Euler for the integral term
    and a first-order derivative filter with time constant ``T_f``.

    Parameters
    ----------
    K_p : float
        Proportional gain.
    T_i : float
        Integral time constant [s].  Set to ``float('inf')`` to disable.
    T_d : float
        Derivative time constant [s].  Set to 0 to disable.
    T_f : float
        Derivative filter time constant [s].
        Default: ``T_d / 10`` (assigned automatically if 0).
    dt : float
        Controller sampling interval [s].
    u_min : float
        Lower output limit (anti-windup).
    u_max : float
        Upper output limit (anti-windup).
    """

    def __init__(
        self,
        K_p: float,
        T_i: float,
        T_d: float,
        T_f: float = 0.0,
        dt: float = 1.0,
        u_min: float = 0.0,
        u_max: float = 100.0,
    ) -> None:
        self.K_p = K_p
        self.T_i = T_i
        self.T_d = T_d
        self.T_f = T_f if T_f > 0 else max(T_d / 10.0, dt)
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

        self._integral: float = 0.0
        self._e_prev: float = 0.0
        self._deriv_filter: float = 0.0

    def reset(self) -> None:
        """Reset internal state."""
        self._integral = 0.0
        self._e_prev = 0.0
        self._deriv_filter = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute the controller output for one time step.

        Parameters
        ----------
        setpoint : float
            Reference signal.
        measurement : float
            Current plant output.

        Returns
        -------
        float
            Saturated controller output.
        """
        e = setpoint - measurement

        # Proportional
        p = self.K_p * e

        # Integral (backward Euler) with anti-windup clamping
        i = self._integral + self.K_p / self.T_i * e * self.dt

        # Derivative with first-order filter
        alpha = self.T_f / (self.T_f + self.dt)
        raw_deriv = self.K_p * self.T_d * (e - self._e_prev) / self.dt
        d = alpha * self._deriv_filter + (1.0 - alpha) * raw_deriv

        u_unsat = p + i + d

        # Output saturation
        u = max(self.u_min, min(self.u_max, u_unsat))

        # Anti-windup: only integrate when not saturated
        if u == u_unsat:
            self._integral = i

        self._e_prev = e
        self._deriv_filter = d

        return u
