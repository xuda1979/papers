# Energy Efficient AI - Experiment Runner
# This script reproduces all experiments and plots for the paper.

Write-Host "Starting Energy Efficient AI Experiments..." -ForegroundColor Green

# Check for Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run experiments
Write-Host "Running simulation suite (this may take a few minutes)..." -ForegroundColor Yellow
python experiments_v2.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Experiments completed successfully!" -ForegroundColor Green
    Write-Host "Plots generated:"
    Get-ChildItem *.png | Select-Object Name
    Write-Host "Summary log: experiment_summary.txt"
} else {
    Write-Error "Experiments failed with exit code $LASTEXITCODE"
}
