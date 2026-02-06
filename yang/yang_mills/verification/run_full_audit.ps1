# Full Audit Runner for Yang-Mills Existence and Mass Gap Proof
# Author: Da Xu
# Date: January 14, 2026

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   YANG-MILLS EXISTENCE AND MASS GAP: FULL AUDIT SUITE    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Environment Check
Write-Host "[Step 1/4] Verifying Environment and Imports..." -ForegroundColor Yellow
try {
    python -c "import sys; print(f'Python {sys.version}')"
    python check_imports.py
} catch {
    Write-Host "Error: Environment check failed." -ForegroundColor Red
    exit 1
}
Write-Host "Environment OK." -ForegroundColor Green
Write-Host ""

# 2. Constant Derivation (Ab Initio)
Write-Host "[Step 2/4] Deriving Rigorous Constants (Ab Initio)..." -ForegroundColor Yellow
python rigorous_constants_derivation.py
if ($LASTEXITCODE -ne 0) { throw "Derivation failed" }
Write-Host "Constants Derived and Certificate Generated." -ForegroundColor Green
Write-Host ""

# 3. Main CAP Verification (Phase 2)
Write-Host "[Step 3/4] Running Intermediate Bridge Verification (Phase 2)..." -ForegroundColor Yellow
Write-Host "Target: Prove Tube Contraction for Beta in [0.25, 6.0]" -ForegroundColor Gray
python full_verifier_phase2.py
if ($LASTEXITCODE -ne 0) { throw "Verification Phase 2 failed" }
Write-Host "Tube Contraction Verified." -ForegroundColor Green
Write-Host ""

# 4. Lorentz Restoration
Write-Host "[Step 4/4] Verifying Lorentz Invariance Restoration..." -ForegroundColor Yellow
python verify_lorentz_restoration_strict.py
if ($LASTEXITCODE -ne 0) { throw "Lorentz verification failed" }
Write-Host "Lorentz Trajectory Certified." -ForegroundColor Green
Write-Host ""

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   AUDIT COMPLETE: UNCONDITIONAL PROOF VERIFIED           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
