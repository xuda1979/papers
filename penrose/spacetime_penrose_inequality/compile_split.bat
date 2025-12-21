@echo off
REM Batch file to compile the split paper

cd split

echo Compiling main.tex...
pdflatex -interaction=nonstopmode main.tex

if errorlevel 1 (
    echo.
    echo ERROR: First pdflatex run failed!
    pause
    exit /b 1
)

echo.
echo Running bibtex...
bibtex main

echo.
echo Second pdflatex run...
pdflatex -interaction=nonstopmode main.tex

echo.
echo Third pdflatex run (for cross-references)...
pdflatex -interaction=nonstopmode main.tex

if exist main.pdf (
    echo.
    echo ========================================
    echo SUCCESS! PDF created: split\main.pdf
    echo ========================================
    start main.pdf
) else (
    echo.
    echo ERROR: PDF was not created!
)

pause
