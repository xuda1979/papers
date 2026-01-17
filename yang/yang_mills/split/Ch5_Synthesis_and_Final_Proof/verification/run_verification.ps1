# PowerShell script to run the minimal verification
# This script executes the independent verifier against the included certificate.

Write-Host "Starting Independent Verification of Yang-Mills Mass Gap Proof (Intermediate Bridge)..."
Write-Host "-------------------------------------------------------------------------------------"

$python_cmd = "python"
# Check if python3 is used instead
if (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $python_cmd = "python3"
}

$script_path = Join-Path $PSScriptRoot "run_rigorous_verification.py"
# For demonstration, we use the sample certificate. In a full reproduction, this would be the 500MB full certificate.
$cert_path = Join-Path $PSScriptRoot "certificate_rigorous_sample.json"

if (-not (Test-Path $cert_path)) {
    Write-Host "Sample certificate not found. Generating..."
    $gen_script = Join-Path $PSScriptRoot "generate_rigorous_certificate.py"
    & $python_cmd $gen_script
}


if (-not (Test-Path $cert_path)) {
    Write-Error "Certificate file not found: $cert_path"
    exit 1
}

& $python_cmd $script_path --cert $cert_path

if ($LASTEXITCODE -eq 0) {
    Write-Host "-------------------------------------------------------------------------------------"
    Write-Host "VERIFICATION PASSED." -ForegroundColor Green
} else {
    Write-Host "-------------------------------------------------------------------------------------"
    Write-Host "VERIFICATION FAILED." -ForegroundColor Red
    exit $LASTEXITCODE
}
