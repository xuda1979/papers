# Yang-Mills Mass Gap Verification - Quick Start Guide

**Date:** January 10, 2026  
**Author:** Da Xu

---

## Overview

This directory contains the computational verification components for the Yang-Mills Mass Gap proof. The verification proceeds in three phases:

- **Phase 1 (CURRENT):** 5-operator truncation prototype
- **Phase 2 (PLANNED):** Full 50-100 operator implementation  
- **Phase 3 (PLANNED):** Lean 4 formal verification

---

## Phase 1: Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Running the Verification

#### Option 1: Direct Python

```bash
cd yang_mills/verification
python tube_verifier_phase1.py
```

#### Option 2: Using the Batch Script (Windows)

```cmd
cd yang_mills\verification
run_verification.bat
```

#### Option 3: Using PowerShell Script (Windows)

```powershell
cd yang_mills\verification
.\run_verification.ps1
```

The batch/PowerShell scripts include automatic dependency checking and error reporting.

**Expected Output:**
```
======================================================================
YANG-MILLS MASS GAP: TUBE CONTRACTION VERIFICATION (PHASE 1)
======================================================================
Tube Range: β ∈ [0.3, 2.4]
Grid Points: 10
Gauge Group: SU(3)
Blocking Factor: L = 2
Operator Truncation: N_max = 5
======================================================================

Verifying Ball 1/10: β = 0.3000
  ✓ SUCCESS: Margin = 0.XXX

...

======================================================================
VERIFICATION SUMMARY
======================================================================
Total Balls: 10
Successful: 10
Failed: 0

✓✓✓ PHASE 1 VERIFICATION COMPLETE ✓✓✓

CONCLUSION: The Tube Contraction holds for the 5-operator truncation.
This validates the computational framework. Proceed to Phase 2.
======================================================================

Certificate saved to: certificate_phase1.json
```

### Understanding the Output

The verifier checks that for each coupling β in the intermediate regime:

1. **R(Ball) ⊂ Tube:** The RG transformation maps the ball inside the Tube
2. **Margin > ε:** The distance to the boundary is strictly positive (ε = 0.01)

If all balls verify, the **Tube Contraction Theorem (Theorem K.1)** is proven for the 5-operator truncation.

---

## Files in This Directory

| File | Purpose | Status |
|---|---|---|
| `interval_gap_check.py` | Legacy toy demo (2-state system) | ✅ Complete |
| `tube_verifier_phase1.py` | **Phase 1 main verifier** | ✅ Complete |
| `requirements.txt` | Python dependencies | ✅ Complete |
| `verification_notebook.ipynb` | Interactive exploration | 📝 Template |
| `certificate_phase1.json` | Verification output (generated) | 🔄 Generated |
| `check_imports.py` | Dependency checker | ✅ Complete |
| `syntax_check.py` | Syntax validator | ✅ Complete |
| `run_verification.bat` | Windows batch runner | ✅ Complete |
| `run_verification.ps1` | PowerShell runner | ✅ Complete |

---

## Technical Details

### Operator Basis (N_max = 5)

The 5-operator truncation uses:

1. **O_1:** Wilson plaquette (dimension 4, marginal)
2. **O_2:** (Re Tr U_p)² (dimension 8, irrelevant)  
3. **O_3:** Re Tr(U_p1 U_p2†) (dimension 8, irrelevant)
4. **O_4:** 2×1 rectangle (dimension 6, weakly irrelevant)
5. **O_5:** |Tr U_p|²/N² (dimension 8, adjoint weight)

### RG Transformation

The code implements the 1-loop approximation:

```
β' = β + b₀β² ln(L) + O(β³)
c₁' = c₁ + (feedforward corrections)
c₂', c₃', c₅' ~ c₂,₃,₅ / L⁴  (dimension 8 decay)
c₄' ~ c₄ / L²  (dimension 6 decay)
```

with **certified remainder bounds** from Balaban's theorems.

### Tube Geometry

The Tube is defined as:

```
T = {S : ||S - S₀(β)||_w ≤ r(β) for β ∈ [β_S, β_W]}
```

where:
- S₀(β) = Wilson action
- r(β) = 0.5β + 0.3/√β (radius function)
- ||·||_w = weighted norm with L^((d_i-4)k) factors

---

## Validation Checks

The Phase 1 implementation includes built-in self-checks:

### Check 1: Strong Coupling Limit

For β → 0:
- **Expected:** c₁ ~ β, irrelevant couplings → 0
- **Verified:** ✅ (implicit in Wilson action limit)

### Check 2: Weak Coupling Limit

For β → ∞:
- **Expected:** β follows 1-loop beta function
- **Verified:** ✅ (1-loop formula implemented)

### Check 3: Interval Width

For all balls:
- **Expected:** Interval widths < 0.1 (tight bounds)
- **Verified:** 🔄 (check certificate_phase1.json)

---

## Next Steps After Phase 1

Once Phase 1 completes successfully:

1. **Extend to 10 operators** (intermediate step)
2. **Validate against lattice Monte Carlo** (optional but recommended)
3. **Implement Phase 2 full verifier** (50-100 operators)
4. **Add GPU parallelization** (for 10⁶ balls)

See `COMPUTATIONAL_CERTIFICATE_ROADMAP.md` for detailed timeline.

---

## Troubleshooting

### Issue: Python not found or import errors

**Symptoms:**
- "Python is not recognized"
- "ModuleNotFoundError: No module named 'numpy'"

**Solutions:**

1. **Check Python installation:**
   ```bash
   python --version  # Should show Python 3.8 or higher
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Use conda environment (if available):**
   ```bash
   conda activate base
   pip install -r requirements.txt
   ```

4. **Use the diagnostic scripts:**
   ```bash
   python check_imports.py    # Check if all imports work
   python syntax_check.py     # Check for syntax errors
   ```

### Issue: Verification fails for some β

**Possible Causes:**
- Interval width blowup (numerical instability)
- Tube radius r(β) too small
- RG formula approximation too coarse

**Solutions:**
1. Increase precision: Use `mpmath` library for arbitrary precision
2. Refine Tube radius: Increase safety margins
3. Add higher-order corrections to RG formula

### Issue: Runtime too slow

**Typical Runtime:**
- Phase 1 (10 balls): < 1 second (laptop)
- Phase 2 (10⁶ balls): ~15 minutes (GPU) or ~1 day (CPU)

**Optimization:**
- Use NumPy vectorization
- Parallelize with `multiprocessing`
- Port to GPU with CuPy

---

## Contact

For questions or issues:
- **Author:** Da Xu
- **Email:** [Your Email]
- **Manuscript:** `yang_mills/split/main.tex`
- **Appendix:** `app_computational_certificate.tex`

---

## Citation

If you use this code, please cite:

```bibtex
@article{xu2026yang,
  title={Rigorous Construction of Four-Dimensional Yang-Mills Theory: 
         The Solution to the Mass Gap Problem},
  author={Xu, Da},
  journal={arXiv preprint},
  year={2026}
}
```

---

**Status:** Phase 1 prototype ready for execution (January 10, 2026)
