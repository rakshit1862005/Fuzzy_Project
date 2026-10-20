# Glass-Box Insight: Project Understanding and Viva Guide

A complete, beginner-friendly guide to **what we built, why, how it works, what it
achieves and what it does not**, written from the actual code, data and results
in this repository.

**In a hurry?** Read the two-page [Quick revision sheet](../QUICK_REVISION.md) first.

## How to use this guide

- Read Chapters 1–2 first: they need no technical background.
- Chapter 3 teaches every concept from scratch; skip what you know.
- Chapters 5–9 explain the method, the maths, the data, the code and the evaluation.
- Chapter 12 traces one real patient through the whole system: if you can explain
  that chapter, you can explain the project.
- Chapters 13–15 are for viva preparation.

Evidence labels used throughout: ✅ **verified** against code, data or results ·
🔎 **interpretation** · ❓ **unknown / not established by the materials**.

## Contents

| # | Chapter | What you get |
|---|---|---|
| 1 | [Executive overview](01-overview.md) | plain-English summary, one-minute pitch, analogy, block diagram |
| 2 | [Problem statement and its real meaning](02-problem-statement.md) | line-by-line reading; proposal vs implementation; scope |
| 3 | [Background concepts from scratch](03-background.md) | ML basics, fuzzy logic, neural networks, neuro-fuzzy, ANFIS |
| 4 | [Existing approaches and why ours](04-existing-approaches.md) | alternatives, comparison table, research context, references |
| 5 | [Architecture](05-architecture.md) | components, data flow, ideal vs implemented |
| 6 | [Mathematics](06-mathematics.md) | every equation with symbols, intuition, numbers and code location |
| 7 | [Data and preprocessing](07-data.md) | columns, missing values, splits, leakage, class balance |
| 8 | [Code walkthrough](08-code-walkthrough.md) | every module in execution order; one patient traced through the code; how to run |
| 9 | [Training and evaluation](09-training-evaluation.md) | training procedure, metrics, baselines, significance, reproducibility |
| 10 | [Traceability](10-traceability.md) | problem → mechanism → code → evidence → limitation |
| 11 | [Results and critical analysis](11-results-critique.md) | every experiment, weaknesses, prioritised improvements |
| 12 | [End-to-end worked example](12-worked-example.md) | patient #152, every number real |
| 13 | [Viva questions and answers](13-viva-questions.md) | 5 levels, project-specific |
| 14 | [Speaking guide](14-speaking-guide.md) | 30 s / 2 min / 5 min scripts, common mistakes |
| 15 | [Revision sheets](15-revision-sheets.md) | summary, glossary, equations, code map, claims to verify |

## Sources of truth

- Code: `glassbox/`, `scripts/`, `webapp/` · Data: `data/diabetes.csv`
- Results: `results/summary.md`, `results/rules.md`, `results/case_studies.md`
  (regenerate with `python scripts/run_experiments.py`)
- Figures in `docs/guide/figures/`: regenerate with `python scripts/make_guide_figures.py`
- Values computed specifically for this guide (not produced by the scripts):
  the McNemar test (Chapter 9), the trivial baselines 64.9 % / 67.5 %, and the
  per-patient network internals (Chapter 12). Each is marked where it appears.

## Relation to the other documents

[`docs/PROBLEM_STATEMENT.md`](../PROBLEM_STATEMENT.md) (revised statement),
[`docs/METHODOLOGY.md`](../METHODOLOGY.md) (report-style method) and
[`docs/VIVA_GUIDE.md`](../VIVA_GUIDE.md) (demo script and short Q&A) remain valid.
This guide goes deeper and adds the critical analysis. Where they differ, this
guide states which is correct.
