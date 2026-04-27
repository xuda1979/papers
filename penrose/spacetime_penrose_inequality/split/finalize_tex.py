import re
import glob

bs = chr(92)
nl = chr(10)

def fix_broken_refs(text):
    # Fix instances where backslash is missing before ref/label/eqref
    text = re.sub(r'(?<!' + bs + r')ref\{', bs + 'ref{', text)
    text = re.sub(r'(?<!' + bs + r')label\{', bs + 'label{', text)
    text = re.sub(r'(?<!' + bs + r')eqref\{', bs + 'eqref{', text)
    
    # Rejoin split references like ef{thm:GapClosed} split across lines
    # Pattern: ef{ followed by newline, then the rest
    text = re.sub(bs + r'ref\{\s*
\s*', bs + 'ref{', text)
    
    # Specifically target the problematic intro file patterns observed in grep
    # e.g., ef{subsec:SpectralGapClosure} (Theorem~

ef{thm:GapClosed})
    # We want to catch the specific sequence of ef{...} (Theorem~

ef{...})
    # and also ef{thm:GapClosed} split as ef{

thm:GapClosed}
    
    return text

for file in glob.glob("*.tex"):
    with open(file, "r") as f:
        content = f.read()
    
    new_content = fix_broken_refs(content)
    
    # Fix the specific multiline split in intro observed
    if file == "sec_01_introduction.tex":
        # Handle the split: ef{subsec:SpectralGapClosure} (Theorem~

ef{thm:GapClosed})
        new_content = re.sub(bs + r'ref\{subsec:SpectralGapClosure\}\s*\(Theorem\s*~\s*
+\s*' + bs + r'ref\{thm:GapClosed\}\)', 
                            bs + 'ref{subsec:SpectralGapClosure} (Theorem~' + bs + 'ref{thm:GapClosed})', new_content)
        
        # Handle the split: Conjecture~

ef{thm:GapClosed}
        new_content = re.sub(r'Conjecture\s*~\s*
+\s*' + bs + r'ref\{thm:GapClosed\}', 
                            'Conjecture~' + bs + 'ref{thm:GapClosed}', new_content)

        # Handle Theorem~

ef{thm:GapClosed}
        new_content = re.sub(r'Theorem\s*~\s*
+\s*' + bs + r'ref\{thm:GapClosed\}', 
                            'Theorem~' + bs + 'ref{thm:GapClosed}', new_content)

    if new_content != content:
        with open(file, "w") as f:
            f.write(new_content)

