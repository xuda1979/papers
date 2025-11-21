#!/usr/bin/env pwsh
# Quick-Start: Generate High-Resolution Data and Update Figures
#
# This script:
# 1. Generates publication-quality high-resolution data (200-300 points per figure)
# 2. Creates backup of current paper.tex
# 3. Provides instructions for updating LaTeX
#
# Usage: .\enhance_figures.ps1

Write-Host "========================================" -ForegroundColor Green
Write-Host "FIGURE ENHANCEMENT QUICK-START" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "paper.tex")) {
    Write-Host "ERROR: paper.tex not found!" -ForegroundColor Red
    Write-Host "Please run this script from the black_hole output directory." -ForegroundColor Red
    exit 1
}

# Step 1: Create backup
Write-Host "[1/4] Creating backup of paper.tex..." -ForegroundColor Cyan
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "paper.tex" "paper.tex.backup_$timestamp"
Write-Host "      Backup saved as: paper.tex.backup_$timestamp" -ForegroundColor Gray

# Step 2: Generate high-resolution data
Write-Host ""
Write-Host "[2/4] Generating high-resolution data..." -ForegroundColor Cyan
if (Test-Path "generate_highres_data.py") {
    python generate_highres_data.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      ✓ Data generation successful!" -ForegroundColor Green
    } else {
        Write-Host "      ✗ Data generation failed!" -ForegroundColor Red
        Write-Host "      Check that Python and NumPy are installed." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "      ERROR: generate_highres_data.py not found!" -ForegroundColor Red
    exit 1
}

# Step 3: List generated files
Write-Host ""
Write-Host "[3/4] Generated high-resolution data files:" -ForegroundColor Cyan
Get-ChildItem "*_highres.dat" | ForEach-Object {
    $lines = (Get-Content $_.Name | Measure-Object -Line).Lines
    Write-Host "      ✓ $($_.Name) - $lines data points" -ForegroundColor Gray
}

# Step 4: Provide update instructions
Write-Host ""
Write-Host "[4/4] Next steps to update your paper:" -ForegroundColor Cyan
Write-Host ""
Write-Host "OPTION A: Quick automatic replacement (simple figures)" -ForegroundColor Yellow
Write-Host "  1. Find: \datatablePagecurve" -ForegroundColor Gray
Write-Host "     Replace with: pagecurve_mem4_highres" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Find: \datatableGtwo" -ForegroundColor Gray
Write-Host "     Replace with: g2_comprehensive_highres" -ForegroundColor Gray
Write-Host ""

Write-Host "OPTION B: Manual enhancement (recommended for publication)" -ForegroundColor Yellow
Write-Host "  See LATEX_EXAMPLES_COMPLEX_FIGURES.md for:" -ForegroundColor Gray
Write-Host "    - Multi-curve plots with confidence bands" -ForegroundColor Gray
Write-Host "    - Parameter sweep heatmaps" -ForegroundColor Gray
Write-Host "    - Multi-panel figures" -ForegroundColor Gray
Write-Host "    - Professional annotations and styling" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "KEY IMPROVEMENTS YOU'LL GET:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✓ 200-300 data points (was 12-22)" -ForegroundColor Green
Write-Host "  ✓ Smooth, professional curves" -ForegroundColor Green
Write-Host "  ✓ Multiple scenarios per figure" -ForegroundColor Green
Write-Host "  ✓ Confidence intervals (shaded bands)" -ForegroundColor Green
Write-Host "  ✓ Rich annotations and markers" -ForegroundColor Green
Write-Host "  ✓ Publication-quality appearance" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Green
Write-Host "READY TO UPDATE LATEX!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Read: ENHANCE_FIGURES_GUIDE.md for detailed instructions" -ForegroundColor Cyan
Write-Host "Examples: LATEX_EXAMPLES_COMPLEX_FIGURES.md" -ForegroundColor Cyan
Write-Host ""
