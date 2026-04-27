"""
Simulation package for the Kiwitsche 2-Punkt-Verfahren.

Public API
----------
BangBangController
    Two-point controller with hysteresis.
PIDController
    Discrete PID with anti-windup.
FOPDTPlant
    First-order-plus-dead-time plant model.
SOPDTPlant
    Second-order-plus-dead-time plant model.
NonlinearFOPDTPlant
    FOPDT-like plant with operating-point-dependent gain and time constant.
simulate_bang_bang
    Run a Bang-Bang closed-loop simulation.
"""

from simulation.bang_bang_control import BangBangController, simulate_bang_bang
from simulation.pid_controller import PIDController
from simulation.plant_models import FOPDTPlant, SOPDTPlant, NonlinearFOPDTPlant

__all__ = [
    "BangBangController",
    "simulate_bang_bang",
    "PIDController",
    "FOPDTPlant",
    "SOPDTPlant",
]
