# PowerShell script to run Phase 1 verification
# Run with: .\run_verification.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Yang-Mills Phase 1 Verification Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ or activate your conda environment" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$output = python check_imports.py 2>&1
Write-Host $output
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Missing dependencies" -ForegroundColor Red
    Write-Host "Run: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Checking syntax..." -ForegroundColor Yellow
$output = python syntax_check.py 2>&1
Write-Host $output
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Syntax errors found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Phase 1 Verification..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python tube_verifier_phase1.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Verification failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ Verification Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}
