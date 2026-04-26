# Contributing to Das Kiwitsche 2-Punkt-Verfahren

Thank you for your interest in contributing. This is a scientific publication
repository managed primarily by AI agents (GitHub Copilot Coding Agent). Human
contributions are equally welcome.

---

## Before You Start

1. Read [`AI_CONTEXT.md`](AI_CONTEXT.md) — it describes the architecture,
   conventions, and scientific background in detail.
2. Check the open issues and the roadmap in `AI_CONTEXT.md` before starting
   new work.

---

## Commit Messages

Follow the conventional-commits style:

```
<type>(<scope>): <short description>

[optional body]
```

| Type | Use for |
|------|---------|
| `feat` | New content (new section, new simulation) |
| `fix` | Corrections to existing content or code |
| `docs` | Documentation / README / AI_CONTEXT changes |
| `style` | Formatting, whitespace (no logic change) |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or fixing tests |
| `chore` | Tooling, CI, dependency updates |

Examples:

```
feat(paper): add derivation of FOPDT identification formulas
fix(simulation): correct dead-time discretisation in plant model
docs(AI_CONTEXT): add note on figure naming convention
```

---

## Pull Requests

- Target branch: `main`
- Title should follow the same `type(scope): description` format.
- Include a brief description of *what* changed and *why*.
- For paper changes, note which sections are affected.
- For Python changes, confirm that simulation scripts run and figures are
  generated without errors.

---

## Python Code Style

- Follow PEP 8.
- Type hints on all public functions.
- Docstrings on all public functions and classes (NumPy docstring style).
- Run `flake8 python/` before committing.

## LaTeX Style

- Keep lines ≤ 100 characters where possible.
- Use `\SI{value}{unit}` (from the `siunitx` package) for all quantities with
  units.
- Labels: `fig:<name>`, `tab:<name>`, `eq:<name>`, `sec:<name>`.

---

## Questions

Open a GitHub issue with the label `question`.
