# Troubleshooting Guide for Yang-Mills Verification

**Date:** January 10, 2026

---

## Common Issues and Solutions

### 1. Python Not Found

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solutions:**

#### A. Check if Python is installed
```bash
# Try different commands
python --version
python3 --version
py --version
```

#### B. Add Python to PATH (Windows)
1. Find Python installation directory (e.g., `C:\Python310` or `C:\Users\YourName\Anaconda3`)
2. Add to system PATH:
   - Right-click "This PC" → Properties → Advanced system settings
   - Environment Variables → System variables → Path → Edit
   - Add Python directory and Scripts subdirectory
3. Restart terminal

#### C. Use full path
```bash
# Replace with your actual Python path
C:\Python310\python.exe tube_verifier_phase1.py
```

#### D. Use Anaconda/Miniconda
```bash
# Open Anaconda Prompt
conda activate base
cd path\to\verification
python tube_verifier_phase1.py
```

---

### 2. Import Errors (NumPy not found)

**Error:**
```
ModuleNotFoundError: No module named 'numpy'
```

**Solutions:**

#### A. Install dependencies
```bash
pip install -r requirements.txt
```

#### B. Check which Python/pip you're using
```bash
python --version    # Should match pip's Python
pip --version       # Should point to same Python
```

If they don't match:
```bash
python -m pip install -r requirements.txt
```

#### C. Create a virtual environment (recommended)
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Syntax Errors

**Error:**
```
SyntaxError: invalid syntax
```

**Solutions:**

#### A. Check Python version
The code requires Python 3.7+ (for dataclasses).

```bash
python --version  # Should be 3.7 or higher
```

If version is too old:
- Install newer Python
- Or use conda: `conda install python=3.10`

#### B. Run syntax checker
```bash
python syntax_check.py
```

This will show exactly where any syntax errors are.

---

### 4. Verification Script Runs but Shows No Output

**Issue:** Script seems to hang or shows nothing

**Solutions:**

#### A. Run with explicit output
```bash
python tube_verifier_phase1.py 2>&1 | Tee-Object -FilePath output.log
```

#### B. Check if it's actually running
```bash
# Open Task Manager (Windows) or top (Linux)
# Look for python.exe process
```

#### C. Add debug output
Edit `tube_verifier_phase1.py` and add at the beginning of `main()`:
```python
def main():
    print("DEBUG: Starting verification...")
    import sys
    sys.stdout.flush()
    # ... rest of code
```

---

### 5. NumPy Version Issues

**Error:**
```
AttributeError: module 'numpy' has no attribute 'finfo'
```

**Solution:**

Update NumPy:
```bash
pip install --upgrade numpy
```

Required version: NumPy >= 1.20.0

---

### 6. Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'certificate_phase1.json'
```

**Solutions:**

#### A. Close any programs that might have the file open
- Close Excel, text editors, etc.

#### B. Run with admin privileges
- Right-click PowerShell/Command Prompt → "Run as Administrator"

#### C. Change output directory
Edit the script to save to a different location:
```python
output_file = r'C:\Users\YourName\Documents\certificate_phase1.json'
```

---

### 7. Interval Width Explosion

**Issue:** Verification fails because interval bounds are too wide

**Solutions:**

#### A. Check the certificate file
```bash
python -c "import json; print(json.load(open('certificate_phase1.json'))['results'][0])"
```

#### B. Increase precision
This is a known issue with naive interval arithmetic. Solutions:
1. Use higher-precision arithmetic (will be in Phase 2)
2. Reduce the ball radius in `verify_ball()`
3. Increase the number of Taylor expansion terms

---

### 8. Conda Environment Issues

**Error:**
```
CondaError: Run 'conda init' before 'conda activate'
```

**Solutions:**

#### A. Initialize conda for your shell
```bash
# PowerShell
conda init powershell

# Command Prompt
conda init cmd.exe

# Then restart terminal
```

#### B. Use conda run instead
```bash
conda run -n base python tube_verifier_phase1.py
```

---

## Diagnostic Workflow

Follow these steps in order:

### Step 1: Check Python
```bash
python check_imports.py
```
**Expected:** All imports show ✓

**If failed:** See sections 1-2 above

### Step 2: Check Syntax
```bash
python syntax_check.py
```
**Expected:** "Syntax check passed successfully!"

**If failed:** See section 3 above

### Step 3: Run Simple Test
```bash
python -c "from tube_verifier_phase1 import Interval; print(Interval(1,2))"
```
**Expected:** `[1.000000e+00, 2.000000e+00]`

**If failed:** Check imports and Python version

### Step 4: Run Full Verification
```bash
python tube_verifier_phase1.py
```
**Expected:** Verification summary showing success

---

## Platform-Specific Notes

### Windows

- Use backslashes in paths: `C:\path\to\file`
- Or use raw strings: `r'C:\path\to\file'`
- PowerShell may require: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Linux/Mac

- Use forward slashes: `/path/to/file`
- May need `python3` instead of `python`
- May need `pip3` instead of `pip`

---

## Getting Help

If none of the above solutions work:

1. **Collect diagnostic information:**
   ```bash
   python --version > diagnostic.txt
   pip list >> diagnostic.txt
   python check_imports.py >> diagnostic.txt
   python syntax_check.py >> diagnostic.txt
   ```

2. **Check the error message carefully:**
   - What is the exact error?
   - What line number?
   - What was the last successful output?

3. **Simplify the problem:**
   - Can you import the modules individually?
   - Does `check_imports.py` work?
   - Does a simple "Hello World" Python script work?

---

## Quick Reference

| Problem | Command |
|---|---|
| Check Python | `python --version` |
| Install dependencies | `pip install -r requirements.txt` |
| Check imports | `python check_imports.py` |
| Check syntax | `python syntax_check.py` |
| Run verification | `python tube_verifier_phase1.py` |
| Use batch runner | `run_verification.bat` |
| Use PowerShell runner | `.\run_verification.ps1` |

---

## Contact

If you encounter an issue not covered here, please:
1. Check the main README.md
2. Review the code comments in tube_verifier_phase1.py
3. Consult the manuscript Appendix 23 for theoretical details

---

**Last Updated:** January 10, 2026
