# Das Kiwitsche 2-Punkt-Verfahren

**A fully AI-managed scientific repository.**

A paper on a systematic procedure for estimating optimal PID control parameters using Bang-Bang (two-point) control as a starting point — the *Kiwitsche 2-Punkt-Verfahren*.

---

## Overview

Classical PID tuning methods (Ziegler–Nichols, Cohen–Coon, etc.) rely on step responses or manual tuning, which can be time-consuming and impractical for complex industrial systems. This paper presents the **Kiwitsche 2-Punkt-Verfahren**: a procedure that leverages the inherent Bang-Bang (on/off) oscillation of a system to extract the plant characteristics needed to compute optimal PID parameters automatically.

The method is especially suited for:

- Industrial cooling/HVAC systems
- Thermal processes with significant dead time
- Any SISO plant that can safely be excited by two-point (Bang-Bang) control

---

## Repository Structure

```
.
├── paper/                   # LaTeX source for the publication
│   ├── main.tex             # Master LaTeX document
│   ├── bibliography.bib     # BibTeX references
│   ├── sections/            # Individual paper sections
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── theory.tex
│   │   ├── methodology.tex
│   │   ├── results.tex
│   │   ├── nonlinear_extension.tex
│   │   ├── discussion.tex
│   │   └── conclusion.tex
│   └── figures/             # Generated figures (output of Python scripts)
│
├── python/                  # Python source for simulation, analysis, plotting
│   ├── requirements.txt
│   ├── simulation/          # Plant models and controller implementations
│   │   ├── bang_bang_control.py
│   │   ├── pid_controller.py
│   │   └── plant_models.py
│   ├── analysis/            # Parameter estimation and data evaluation
│   │   └── parameter_estimation.py
│   └── plotting/            # Figure generation for the paper
│       └── plot_results.py
│
├── AI_CONTEXT.md            # Context and conventions for AI agents / future contributors
├── CONTRIBUTING.md          # Contribution guide
└── README.md
```

---

## Quickstart

### Build the Paper (requires a LaTeX distribution, e.g. TeX Live)

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Run Python Simulations

```bash
cd python
pip install -r requirements.txt
python -m simulation.bang_bang_control   # run Bang-Bang demo
python -m analysis.parameter_estimation  # estimate PID parameters
python -m plotting.plot_results          # generate figures for the paper
```

Generated figures are saved to `paper/figures/`.

---

## Method Summary

The *Kiwitsche 2-Punkt-Verfahren* proceeds in four steps:

1. **Bang-Bang Excitation** — Apply two-point (on/off) control to the closed loop. The system naturally settles into a limit-cycle oscillation.
2. **Characteristic Extraction** — From the oscillation, measure the ultimate period *T_u* and the peak-to-peak amplitude *A_u*.
3. **Plant Identification** — Use *T_u* and *A_u* together with knowledge of the Bang-Bang switching levels to derive an equivalent first-order-plus-dead-time (FOPDT) model.
4. **PID Synthesis** — Apply IMC-based tuning formulas to the FOPDT model to compute *K_p*, *T_i*, *T_d*, using the estimated dead time as the closed-loop tuning parameter *λ*.

A detailed derivation, stability analysis, and experimental validation are provided in the paper.

---

## AI-Managed Repository

This repository is fully managed by AI (GitHub Copilot Coding Agent). See [`AI_CONTEXT.md`](AI_CONTEXT.md) for conventions, design decisions, and guidance for future AI agents or human developers who wish to branch or extend this work.

---

## License

To be determined upon publication. Until then, all rights reserved.
