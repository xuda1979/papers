@echo off
setlocal EnableExtensions

REM Perl-free build fallback for MiKTeX/TeX Live on Windows.
REM Uses pdflatex + biber (this project uses biblatex with backend=biber).
REM
REM Usage:
REM   build_paper_nolatexmk.bat
REM
REM Output: main.pdf (in this directory)

set TEX=pdflatex
set BIB=biber
set JOB=main
set MAIN=%JOB%.tex

echo === Pass 1: %TEX% %MAIN% ===
%TEX% -interaction=nonstopmode -file-line-error -synctex=1 %MAIN%
if errorlevel 1 goto :fail

echo === Bibliography: %BIB% %JOB% ===
%BIB% %JOB%
if errorlevel 1 goto :fail

echo === Pass 2: %TEX% %MAIN% ===
%TEX% -interaction=nonstopmode -file-line-error -synctex=1 %MAIN%
if errorlevel 1 goto :fail

echo === Pass 3: %TEX% %MAIN% ===
%TEX% -interaction=nonstopmode -file-line-error -synctex=1 %MAIN%
if errorlevel 1 goto :fail

echo === Done. Built %JOB%.pdf ===
goto :eof

:fail
echo.
echo Build failed. Check %JOB%.log for details.
exit /b 1
