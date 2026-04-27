"""
Plant models for KZV simulations.

All models are implemented as callable objects with a `step` method for
time-domain simulation using the Euler/ZOH discretisation.

Units
-----
Time  : seconds  [s]
Input : percent  [%]  (0–100)
Output: degrees Celsius [°C]  (or generic dimensionless for non-thermal plants)
"""

from __future__ import annotations

import numpy as np


class FOPDTPlant:
    """First-Order-Plus-Dead-Time (FOPDT) plant.

    Parameters
    ----------
    K : float
        Process gain [output_unit / input_unit].
    T : float
        Time constant [s].
    L : float
        Dead time [s].
    dt : float
        Simulation time step [s].
    y0 : float
        Initial output value.
    """

    def __init__(
        self,
        K: float,
        T: float,
        L: float,
        dt: float = 0.1,
        y0: float = 0.0,
    ) -> None:
        self.K = K
        self.T = T
        self.L = L
        self.dt = dt

        # Dead-time buffer
        self._delay_steps = max(1, int(round(L / dt)))
        self._u_buffer: list[float] = [0.0] * self._delay_steps

        # State
        self.y = y0

    def step(self, u: float) -> float:
        """Advance the plant by one time step.

        Parameters
        ----------
        u : float
            Current control input [%].

        Returns
        -------
        float
            Plant output after this time step [°C].
        """
        # Get delayed input from buffer
        u_delayed = self._u_buffer.pop(0)
        self._u_buffer.append(u)

        # Euler integration of first-order ODE: T * dy/dt + y = K * u
        dydt = (self.K * u_delayed - self.y) / self.T
        self.y += dydt * self.dt
        return self.y


class SOPDTPlant:
    """Second-Order-Plus-Dead-Time (SOPDT) plant.

    Transfer function:
        G(s) = K * exp(-L*s) / ((T1*s + 1) * (T2*s + 1))

    Parameters
    ----------
    K : float
        Process gain.
    T1 : float
        First time constant [s].
    T2 : float
        Second time constant [s].
    L : float
        Dead time [s].
    dt : float
        Simulation time step [s].
    y0 : float
        Initial output value.
    """

    def __init__(
        self,
        K: float,
        T1: float,
        T2: float,
        L: float,
        dt: float = 0.1,
        y0: float = 0.0,
    ) -> None:
        self.K = K
        self.T1 = T1
        self.T2 = T2
        self.L = L
        self.dt = dt

        # Dead-time buffer
        self._delay_steps = max(1, int(round(L / dt)))
        self._u_buffer: list[float] = [0.0] * self._delay_steps

        # Two first-order states
        self._x1 = y0  # output of first lag
        self._x2 = y0  # output of second lag (plant output)

    @property
    def y(self) -> float:
        """Current plant output."""
        return self._x2

    def step(self, u: float) -> float:
        """Advance the plant by one time step.

        Parameters
        ----------
        u : float
            Control input.

        Returns
        -------
        float
            Plant output.
        """
        u_delayed = self._u_buffer.pop(0)
        self._u_buffer.append(u)

        dx1 = (self.K * u_delayed - self._x1) / self.T1
        dx2 = (self._x1 - self._x2) / self.T2
        self._x1 += dx1 * self.dt
        self._x2 += dx2 * self.dt
        return self._x2


class NonlinearFOPDTPlant:
    """FOPDT-like plant whose gain and time constant vary with operating point.

    This models a class of mildly nonlinear processes (e.g.\ cooling systems
    with heat-transfer coefficients that depend on temperature) by making
    :math:`K` and :math:`T` smooth functions of the current output:

    .. math::
        K_{\\text{eff}}(y) &= K_0 \\cdot (1 + \\alpha_K \\cdot (y - y_{\\text{ref}})) \\\\
        T_{\\text{eff}}(y) &= T_0 \\cdot (1 + \\alpha_T \\cdot (y - y_{\\text{ref}}))

    The dead time :math:`L` is assumed constant.

    Parameters
    ----------
    K0 : float
        Nominal process gain at the reference operating point ``y_ref``.
    T0 : float
        Nominal time constant [s] at ``y_ref``.
    L : float
        Dead time [s] (constant).
    dt : float
        Simulation time step [s].
    y0 : float
        Initial output value.
    y_ref : float
        Reference operating point around which the linearisation is defined.
    alpha_K : float
        Gain sensitivity: relative change of K per unit deviation from y_ref.
        Negative means gain decreases as output rises (typical for cooling).
    alpha_T : float
        Time-constant sensitivity: relative change of T per unit deviation.
        Positive means the plant slows down as output rises.
    K_min : float
        Lower bound on K_eff to prevent non-physical values.
    T_min : float
        Lower bound on T_eff [s].
    """

    def __init__(
        self,
        K0: float,
        T0: float,
        L: float,
        dt: float = 0.1,
        y0: float = 0.0,
        y_ref: float = 20.0,
        alpha_K: float = -0.01,
        alpha_T: float = 0.005,
        K_min: float = 0.05,
        T_min: float = 5.0,
    ) -> None:
        self.K0 = K0
        self.T0 = T0
        self.L = L
        self.dt = dt
        self.y_ref = y_ref
        self.alpha_K = alpha_K
        self.alpha_T = alpha_T
        self.K_min = K_min
        self.T_min = T_min

        self._delay_steps = max(1, int(round(L / dt)))
        self._u_buffer: list[float] = [0.0] * self._delay_steps
        self.y = y0

    def K_eff(self, y: float | None = None) -> float:
        """Effective process gain at current (or given) output value."""
        y_op = self.y if y is None else y
        return max(self.K_min, self.K0 * (1.0 + self.alpha_K * (y_op - self.y_ref)))

    def T_eff(self, y: float | None = None) -> float:
        """Effective time constant at current (or given) output value [s]."""
        y_op = self.y if y is None else y
        return max(self.T_min, self.T0 * (1.0 + self.alpha_T * (y_op - self.y_ref)))

    def step(self, u: float) -> float:
        """Advance the plant by one time step.

        Parameters
        ----------
        u : float
            Control input.

        Returns
        -------
        float
            Plant output after this step.
        """
        u_delayed = self._u_buffer.pop(0)
        self._u_buffer.append(u)

        k = self.K_eff()
        tau = self.T_eff()
        dydt = (k * u_delayed - self.y) / tau
        self.y += dydt * self.dt
        return self.y
