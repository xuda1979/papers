@echo off
REM Batch file to run Phase 1 verification with error handling

echo ========================================
echo Yang-Mills Phase 1 Verification Runner
echo ========================================
echo.

REM Try to activate conda base environment
echo Attempting to activate conda base environment...
call conda activate base 2>nul
if errorlevel 1 (
    echo Warning: Could not activate conda environment
    echo Attempting to use system Python...
)

echo.
echo Checking Python availability...
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ or activate your conda environment
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python check_imports.py
if errorlevel 1 (
    echo ERROR: Missing dependencies
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Checking syntax...
python syntax_check.py
if errorlevel 1 (
    echo ERROR: Syntax errors found
    pause
    exit /b 1
)

echo.
echo ========================================
echo Running Phase 1 Verification...
echo ========================================
echo.

python tube_verifier_phase1.py
if errorlevel 1 (
    echo.
    echo ERROR: Verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Verification Complete!
echo ========================================
pause
