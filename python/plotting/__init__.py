"""
Plotting package for the Kiwitsche 2-Punkt-Verfahren.

Public API
----------
plot_limit_cycle
    Generate the Bang-Bang limit-cycle figure (Figure 1 in the paper).
plot_step_response_comparison
    Generate the closed-loop step-response comparison (Figure 2).
main
    Generate all figures.
"""

from plotting.plot_results import main, plot_limit_cycle, plot_step_response_comparison

__all__ = ["main", "plot_limit_cycle", "plot_step_response_comparison"]
