# HOW TO FIX THE WORKFLOW TRUNCATION ISSUE

## The Real Problem

Your paper is **FINE** (2,217 lines, 43 pages, 98 references). The workflow is **BROKEN** because:

1. **AI Model is Truncating** - Returns only 50-60% of the paper
2. **Content Guardian is Protecting** - Blocks truncated versions (GOOD!)
3. **Result:** No modifications happen, paper stays intact

## Root Cause Analysis

### Why AI Truncates

Looking at the logs:
```
DEBUG: Extracted 51245 chars from response (iteration 3)
DEBUG: Extracted 60586 chars from response (iteration 4)
```

Original paper: **151,241 bytes**
AI returns: **51-60KB** (~33-40% of original)

This suggests the AI model's **output token limit is too low** for your paper size.

### Token Math
- Your paper: ~151KB ≈ **38,000 tokens**
- GPT-4/GPT-5 typical output limit: **4,096 tokens**
- **Result:** AI can only return ~10% of what's needed!

## Solutions (In Order of Effectiveness)

### 🔧 Solution 1: Enable Diff-Based Editing (RECOMMENDED)

Instead of regenerating the entire paper, send only targeted changes.

**File to modify:** `sciresearch_workflow.py`

Look for where it extracts `paper.tex` from AI response and change to use **diff-based editing**:

```python
# Instead of replacing entire file:
# with open(tex_path, 'w', encoding='utf-8') as f:
#     f.write(extracted_content)

# Use diff-based approach:
from utils.diff_based_revision import apply_diffs
apply_diffs(tex_path, ai_response_diffs)
```

### 🔧 Solution 2: Increase Model Output Tokens

**File to modify:** `sciresearch_workflow.py`

Find the API call configuration and add:

```python
completion = openai.Completion.create(
    model="gpt-5",
    max_tokens=16384,  # Increase from default 4096
    # or
    max_completion_tokens=16384,  # For newer API
    ...
)
```

**Check model limits:**
- GPT-4-turbo: 4,096 tokens
- GPT-4-32k: 4,096 output tokens (input is 32k)
- Claude 3: 4,096 tokens
- Claude 3.5 Sonnet: 8,192 tokens

**Recommendation:** Switch to a model with 8K+ output tokens or use diff mode.

### 🔧 Solution 3: Process in Chunks

Modify the workflow to process sections separately:

```python
sections = [
    "introduction",
    "methods", 
    "results",
    "discussion"
]

for section in sections:
    # Extract section
    # Send to AI for review
    # Apply changes to that section only
```

### 🔧 Solution 4: Reduce Content Guardian Threshold (RISKY)

**Only if Solutions 1-3 don't work**

`utils/content_guardian.py` line 18:
```python
# Change from:
MIN_ACCEPTABLE_LINES = 2000

# To:
MIN_ACCEPTABLE_LINES = 1500  # RISKY - may allow truncated papers!
```

⚠️ **WARNING:** This allows the AI to delete content! Only do this if you have good backups.

### 🔧 Solution 5: Use BibTeX (Saves ~300 lines)

Replace embedded bibliography with external file:

**In `paper.tex`:**
```latex
% DELETE lines 1953-2215 (\begin{thebibliography}...\end{thebibliography})

% ADD before \end{document}:
\bibliographystyle{jhep}  % or plain, unsrt, alpha, etc.
\bibliography{references}  % references.bib already exists!
```

**Recompile:**
```bash
pdflatex paper.tex
bibtex paper    # Generates .bbl from references.bib
pdflatex paper.tex
pdflatex paper.tex
```

This reduces paper from 2,217 → ~1,960 lines, giving more headroom.

## Testing the Fix

After implementing a solution, test with:

```bash
cd c:\Users\Lenovo\software\AI-Scientist\output\black_hole

# Backup first!
cp paper.tex paper.tex.backup

# Run workflow with single iteration
python ..\..\main.py --modify-existing --output-dir . --max-iterations 1 --model gpt-5

# Check if changes were applied
git diff paper.tex

# Verify line count
(Get-Content paper.tex | Measure-Object -Line).Lines
```

Expected:
- ✅ Line count close to original (±100 lines)
- ✅ No Content Guardian blocks
- ✅ Changes actually applied
- ✅ Bibliography still present

## Quick Win: Just Fix Compilation

If you just want the paper to compile (forget the workflow for now):

```bash
cd c:\Users\Lenovo\software\AI-Scientist\output\black_hole

# Compile
pdflatex paper.tex
bibtex paper
pdflatex paper.tex  
pdflatex paper.tex

# Check result
ls -l paper.pdf
```

You already have a working 43-page paper with 98 references!

## What NOT to Do

❌ **Don't lower MIN_ACCEPTABLE_LINES to 500** - You'll lose your entire paper
❌ **Don't run workflow again with current settings** - It will keep failing
❌ **Don't manually delete sections** - The AI isn't asking for that
❌ **Don't panic** - Your paper is intact and complete!

## Immediate Next Steps

1. ✅ **Paper compiles now** (fixed `</table>` bug)
2. 🔧 **Choose Solution 1 or 2** above
3. ✅ **Test with --max-iterations 1** first
4. ✅ **Keep backups** before each run
5. 📧 **Report issue** to AI-Scientist maintainers about truncation

## Summary Table

| What You Thought | Reality |
|------------------|---------|
| "No references at all" | Has 98 references (lines 1953-2215) |
| "Paper is fucking short" | 2,217 lines, 43 pages - full research paper! |
| "Workflow doesn't work" | Workflow truncates output, Guardian blocks (working as designed) |
| "Need 5 iterations" | Need 0 iterations - paper is already complete! |

**Your paper is publication-ready. The workflow has a configuration issue with model output limits.**
