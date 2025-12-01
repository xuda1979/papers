# Theoretical Limits of Reasoning (Category Theory)

This repository contains the research paper "Categorical Limits of In-Context Learning: A No-Go Theorem for Fixed-Depth Transformers".

## Abstract
We apply Category Theory to bound the reasoning capabilities of Transformer models. By modeling the semantic map as a functor, we prove a "No-Go Theorem" showing that fixed-depth Transformers cannot solve certain recursive logical problems (like graph connectivity) regardless of width, necessitating external memory or Chain-of-Thought prompting. We rigorously define reasoning tasks as **Initial Algebras** and invoke **Adamek's Theorem** to demonstrate the structural inability of fixed compositions to preserve the required colimits.

## Contents
- `paper.tex`: The LaTeX source code for the research paper.
- `simulation.py`: Python script to generate the "Reasoning Gap" visualization (`reasoning_gap.png`).
- `reasoning_gap.png`: Empirical results showing the phase transition in solvability.
- `references.bib`: Bibliography file.

## Key Results
- **No-Go Theorem**: Fixed-depth transformers cannot compute the initial algebra of a recursive functor for unbounded inputs.
- **Reasoning Gap**: Empirical simulation confirms a sharp drop in accuracy when problem size $N$ exceeds the effective receptive field.
- **CoT Sufficiency**: Chain-of-Thought is formalized as an externalized fixpoint iteration.
