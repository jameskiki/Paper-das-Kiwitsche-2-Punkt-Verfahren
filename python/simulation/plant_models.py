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
