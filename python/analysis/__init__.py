"""
Analysis package for the Kiwitsche 2-Punkt-Verfahren.

Public API
----------
extract_limit_cycle_characteristics
    Measure T_u and A_u from a Bang-Bang limit-cycle time series.
estimate_dead_time_from_limit_cycle
    Estimate L from the slope sign-change after Bang-Bang switching events.
identify_fopdt
    Estimate FOPDT plant parameters (K, T, L) from limit-cycle data.
imc_pid
    Compute IMC-PID tuning parameters from FOPDT model.
"""

from analysis.parameter_estimation import (
    extract_limit_cycle_characteristics,
    estimate_dead_time_from_limit_cycle,
    identify_fopdt_from_transients,
    identify_fopdt,
    imc_pid,
)

__all__ = [
    "extract_limit_cycle_characteristics",
    "estimate_dead_time_from_limit_cycle",
    "identify_fopdt_from_transients",
    "identify_fopdt",
    "imc_pid",
]
