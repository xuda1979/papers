@echo off
setlocal EnableExtensions

REM Perl-free build script: pdflatex only (no latexmk, no biber).
REM This should succeed as long as a current main.bbl is present.

set TEX=pdflatex
set MAIN=main.tex

echo === Pass 1: %TEX% %MAIN% ===
%TEX% -interaction=nonstopmode -file-line-error -synctex=1 %MAIN%
if errorlevel 1 goto :fail

echo === Pass 2: %TEX% %MAIN% ===
%TEX% -interaction=nonstopmode -file-line-error -synctex=1 %MAIN%
if errorlevel 1 goto :fail

echo === Done. Built main.pdf ===
goto :eof

:fail
echo.
echo Build failed. Check main.log for details.
exit /b 1
