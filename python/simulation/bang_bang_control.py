"""
Bang-Bang (two-point) controller implementation and limit-cycle simulation.

Run as a module for a quick demonstration:
    python -m simulation.bang_bang_control
"""

from __future__ import annotations

import numpy as np

from simulation.plant_models import SOPDTPlant  # noqa: E402


# ─── Controller ──────────────────────────────────────────────────────────────

class BangBangController:
    """Two-point controller with symmetric hysteresis.

    The controller switches between ``u_max`` and ``u_min`` depending on the
    sign of the error ``e = setpoint - measurement``, with a dead-band of
    ±``d`` around zero error.

    Parameters
    ----------
    u_max : float
        High output level (e.g. 100 %).
    u_min : float
        Low output level (e.g. 0 %).
    d : float
        Half-width of the hysteresis band [output units of measurement].
    """

    def __init__(
        self,
        u_max: float = 100.0,
        u_min: float = 0.0,
        d: float = 0.3,
    ) -> None:
        self.u_max = u_max
        self.u_min = u_min
        self.d = d
        self._u: float = u_min  # initial output

    @property
    def M(self) -> float:
        """Relay amplitude: (u_max - u_min) / 2."""
        return (self.u_max - self.u_min) / 2.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute the controller output.

        Parameters
        ----------
        setpoint : float
            Reference value.
        measurement : float
            Current plant output.

        Returns
        -------
        float
            Controller output (u_min or u_max).
        """
        e = setpoint - measurement
        if e > self.d:
            self._u = self.u_max
        elif e < -self.d:
            self._u = self.u_min
        # else: keep previous output (hysteresis)
        return self._u


# ─── Simulation ──────────────────────────────────────────────────────────────

def simulate_bang_bang(
    plant: "FOPDTPlant | SOPDTPlant",
    controller: BangBangController,
    setpoint: float,
    t_end: float,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a plant under Bang-Bang control.

    Parameters
    ----------
    plant : FOPDTPlant | SOPDTPlant
        Any plant with a ``step(u) -> float`` method and a ``y`` attribute.
    controller : BangBangController
        Bang-Bang controller instance.
    setpoint : float
        Reference / desired output value.
    t_end : float
        Total simulation time [s].
    dt : float
        Time step [s].

    Returns
    -------
    t : np.ndarray
        Time vector [s].
    y : np.ndarray
        Plant output history.
    u : np.ndarray
        Controller output history.
    """
    n_steps = int(t_end / dt)
    t = np.linspace(0.0, t_end, n_steps)
    y = np.empty(n_steps)
    u = np.empty(n_steps)

    for k in range(n_steps):
        u_k = controller.compute(setpoint, plant.y)
        y_k = plant.step(u_k)
        t[k] = k * dt
        y[k] = y_k
        u[k] = u_k

    return t, y, u


# ─── Demo ────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Quick demonstration: SOPDT plant under Bang-Bang control."""
    dt = 1.0  # [s]

    plant = SOPDTPlant(K=1.5, T1=120.0, T2=30.0, L=20.0, dt=dt, y0=20.0)
    controller = BangBangController(u_max=100.0, u_min=0.0, d=0.3)
    setpoint = 20.0  # [°C]

    t, y, u = simulate_bang_bang(
        plant=plant,
        controller=controller,
        setpoint=setpoint,
        t_end=3000.0,
        dt=dt,
    )

    print(f"Simulation complete: {len(t)} steps, t_end={t[-1]:.1f} s")
    print(f"Output range in last 1000 s: "
          f"[{y[-1000:].min():.3f}, {y[-1000:].max():.3f}] °C")
    print(f"Amplitude A_u ≈ {(y[-1000:].max() - y[-1000:].min()):.3f} °C")


if __name__ == "__main__":
    _demo()
