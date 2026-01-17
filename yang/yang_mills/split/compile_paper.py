import subprocess
import os

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {command}:")
        print(result.stdout)
        print(result.stderr)
        return False
    print("Success.")
    return True

def compile():
    print("Compiling LaTeX document...")
    
    # 1. pdflatex
    if not run_command("pdflatex -interaction=nonstopmode main.tex"):
        return

    # 2. biber
    if not run_command("biber main"):
        print("Biber failed, but continuing (might include warnings)...")

    # 3. pdflatex again
    if not run_command("pdflatex -interaction=nonstopmode main.tex"):
        return

    # 4. pdflatex again for stable refs
    if not run_command("pdflatex -interaction=nonstopmode main.tex"):
        return

    print("Compilation complete. Output: main.pdf")

if __name__ == "__main__":
    compile()
