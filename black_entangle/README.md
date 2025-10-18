# Black Entangle — GR‑Bell Tests near Black Holes

This folder contains an **innovative rewrite** of the *Black Entangle* paper and a minimal, transparent scaffold to reproduce the main theory curves. The key idea is to operationalize ER=EPR‑style intuition into **falsifiable, observer‑side Bell tests** using lensed photon pairs and polarization transport in Kerr spacetimes.

## What’s new (high level)

* **GR‑corrected CHSH**: A closed‑form observable \(S_{\text{GR}}\) that incorporates gravitational redshift, frame dragging, and parallel‑transported polarization bases along null geodesics.
* **Gauge‑invariant measurement map**: A tetrad‑based, screen‑projected polarization operator that lets asymptotic observers compare outcomes without gauge ambiguity.
* **GIEA (Gravitationally Induced Entanglement Asymmetry)**: A simple, unitless observable that upper‑bounds entanglement negativity from asymmetry in lensed time‑of‑flight vs polarization correlations.
* **Falsifiable predictions**: Where and how to look (binary BHs, microquasars, flares near Sgr A*) and what instrument precision is required to exceed the classical CHSH bound after GR corrections.

## Repository layout

```
black_entangle/
├── paper/
│   └── black_entangle.tex         # The manuscript
├── bib/
│   └── references.bib             # Minimal bibliography
├── code/
│   └── simulate_tpchshr.py        # Tiny simulation stub (plots S_GR vs. impact parameter)
├── scripts/
│   └── regenerate.sh              # One-liner to build PDF and re-run the stub
└── Makefile                       # latexmk build + simple run target
```

## Quickstart

**Build the paper (PDF):**
```bash
cd black_entangle
make
```
This runs `latexmk` on `paper/black_entangle.tex` and writes a PDF next to it.

**Re-generate minimal figures/data (demo):**
```bash
cd black_entangle
python3 code/simulate_tpchshr.py --spin 0.6 --bmin 3.0 --bmax 8.0 --samples 40
```
This prints a CSV-like table of `impact_parameter, S_GR` and stores a small TSV under `results/`.

## Conceptual summary

Consider two entangled photons emitted near a compact region (e.g., a transient flare). In a curved, rotating background (Kerr), their polarization vectors are **parallel‑transported along distinct null geodesics** and redshifted before reaching distant detectors. We define:

1. A **local, orthonormal tetrad** at emission and at each detector.
2. A **screen‑projected polarization basis** transported to infinity.
3. An **effective measurement angle** pair \((\tilde{a}, \tilde{b})\) that is the *only* quantity asymptotic observers can compare without gauge artifacts.

With those ingredients, we derive a **GR‑corrected CHSH functional** \(S_{\text{GR}}\) whose classical upper bound persists but whose *settings* include redshift and transport corrections. If measured \(S_{\text{GR}}>2\) under the GR‑corrected settings, then Bell nonlocality is present *despite* curved‑spacetime distortions. The **GIEA** scalar summarizes how geometry degrades (or reveals) entanglement asymmetrically across lensed paths and directly yields a **conservative upper bound** on the state’s negativity in situ.

## Reproducibility notes

* The included `simulate_tpchshr.py` is intentionally lightweight: it **does not** integrate full Kerr geodesics; instead it parameterizes redshift/transport via simple knobs to let readers reproduce the **qualitative** theory curves and sanity‑check inequalities.
* For full fidelity, couple the measurement map in the manuscript to a standard null‑geodesic integrator (e.g., a public Kerr ray‑tracer) and replace the transport/drag *ansätze* with numerically integrated frames.

## Limitations / next steps

* The included stub is **Newtonian‑to‑post‑Newtonian flavored** (fast to run, pedagogical). Replace it with geodesic‑level transport to produce proposal‑grade forecasts for particular instruments (EHT‑band polarimetry, optical/IR interferometry, high‑time‑resolution polarimeters).
* Extend to **finite‑temperature baths** and **Unruh‑like noise** for accelerated detectors; quantify when thermal noise masks \(S_{\text{GR}}\).

## Build dependencies
* `latexmk` (or `pdflatex` run twice + `bibtex`)
* Python 3.9+ (only for the toy stub)

## License
Same license as the repository root. Bibliographic entries are factual metadata; please review publisher licenses if you redistribute paper PDFs.
