@echo off
setlocal

echo Running Stellarator Design Simulation...

rem Prefer the Python launcher if available.
where py >nul 2>nul
if errorlevel 1 goto :use_python
set "PY=py -3"
goto :run_python

:use_python
set "PY=python"

:run_python
%PY% stellarator_design.py
if errorlevel 1 goto :python_failed

if not exist stellarator_plot.png goto :plot_missing

echo Simulation successful.

where pdflatex >nul 2>nul
if errorlevel 1 goto :no_pdflatex

echo Generating PDF...
pdflatex paper.tex
pdflatex paper.tex
echo Done.
goto :done

:python_failed
echo Simulation failed (Python returned an error).
goto :done

:plot_missing
echo Simulation failed. stellarator_plot.png not found.
goto :done

:no_pdflatex
echo pdflatex not found on PATH. Skipping PDF build.
goto :done

:done
pause
