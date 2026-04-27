"""
deploy.py — Build the full paper from scratch.

Steps
-----
1. Regenerate all figures (runs python/plotting/plot_results.py).
2. Compile the LaTeX document:
       pdflatex  →  bibtex  →  pdflatex  →  pdflatex
   (two extra passes to resolve cross-references and bibliography.)

Run from the repo root:
    python deploy.py

Optional flags:
    --no-figures    Skip figure generation, only recompile LaTeX.
    --no-latex      Skip LaTeX compilation, only regenerate figures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
PAPER_DIR = os.path.join(REPO_ROOT, "paper")
MAIN_TEX = "main"  # without .tex


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str, label: str) -> None:
    """Run a subprocess and abort on non-zero exit."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


# ─── Steps ────────────────────────────────────────────────────────────────────

def build_figures() -> None:
    """Regenerate all paper figures by running both plotting modules."""
    _run(
        [sys.executable, "-m", "plotting.plot_intro_visuals"],
        cwd=PYTHON_DIR,
        label="Generate intro figures (intro_duty_cycle, intro_phase_portrait)",
    )
    _run(
        [sys.executable, "-m", "plotting.plot_results"],
        cwd=PYTHON_DIR,
        label="Generate results figures (limit_cycle, step_response, nonlinear)",
    )


def build_latex() -> None:
    """Compile main.tex with pdflatex + bibtex (full 3-pass build)."""
    tex_cmd = ["pdflatex", "-interaction=nonstopmode", MAIN_TEX]
    bib_cmd = ["bibtex", MAIN_TEX]

    _run(tex_cmd, cwd=PAPER_DIR, label="pdflatex  (pass 1/3)")
    _run(bib_cmd, cwd=PAPER_DIR, label="bibtex")
    _run(tex_cmd, cwd=PAPER_DIR, label="pdflatex  (pass 2/3)")
    _run(tex_cmd, cwd=PAPER_DIR, label="pdflatex  (pass 3/3)")

    pdf_path = os.path.join(PAPER_DIR, f"{MAIN_TEX}.pdf")
    if os.path.isfile(pdf_path):
        print(f"\n[OK] PDF written to:  {pdf_path}")
    else:
        print("\n[WARNING] Build finished but PDF not found — check LaTeX log.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the KZV paper (figures + LaTeX).")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip LaTeX compilation.",
    )
    args = parser.parse_args()

    print("=== KZV Paper Deploy ===")

    if not args.no_figures:
        build_figures()
    else:
        print("\n[SKIP] Figure generation.")

    if not args.no_latex:
        build_latex()
    else:
        print("\n[SKIP] LaTeX compilation.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
