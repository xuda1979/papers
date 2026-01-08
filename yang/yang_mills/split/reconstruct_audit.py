import re
import os

input_path = r'yang_mills/split/app204_mathematical_audit_summary.tex'
output_path = r'yang_mills/split/app204_fixed.tex'

def clean_latex_block(text):
    # Remove start/end spaces
    text = text.strip()
    # Remove any truncated table artifacts at the end
    # e.g. \textbf{7. Rotational...
    text = re.sub(r'\\textbf\{\d+\.\s+[A-Za-z].*?$', '', text, flags=re.DOTALL)
    # Remove stray \hline or \end{tabular} at the end
    text = re.sub(r'\\hline\s*$', '', text)
    text = re.sub(r'\\end\{tabular\}\s*$', '', text)
    text = re.sub(r'\\end\{center\}\s*$', '', text)
    return text

def run():
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("File not found.")
        return

    # 1. Extract Derivations
    # Pattern: \subsubsection{Derivation TAG: Title}
    # We allow the capture to go until the next \subsubsection or \section or start of a table row like \textbf{N.
    
    # We split by \subsubsection to get chunks
    chunks = re.split(r'(\\subsubsection\{Derivation [A-Z]+:.*?\})', content)
    
    derivations = {} # Tag -> {title, body}

    # chunks[0] is preamble
    # chunks[1] is header1, chunks[2] is body1
    # chunks[3] is header2, chunks[4] is body2...
    
    for i in range(1, len(chunks), 2):
        header = chunks[i]
        body = chunks[i+1]
        
        # Parse header: \subsubsection{Derivation TAG: Title}
        m = re.match(r'\\subsubsection\{Derivation ([A-Z]+):\s*(.*?)\}', header)
        if m:
            tag = m.group(1)
            title = m.group(2)
            
            # Clean body
            # Stop body at next section items or table artifacts
            # Aggressive cleanup: stop at \section, \subsection, \textbf{\d+.
            # But the detailed content might have \textbf{Goal:}
            
            # We want to capture the "Goal" and the "enumerate" list.
            # Most bodies end with \end{enumerate}.
            
            # Let's find the last \end{enumerate} in the block and cut there?
            # Or assume the block ends when a new "Part" or "Error Table" starts.
            
            # Split body by obvious section breaks
            stop_markers = [r'\\section', r'\\subsection', r'\\begin\{tabular\}', r'\\begin\{longtable\}', r'\\textbf\{\d+\.']
            
            # Find the earliest occurrence of a stop marker *at the start of a line* or distinct block
            # But wait, \textbf{1. ...} might be inside a table which is inside the text?
            # No, the table lines seem to interrupt.
            
            valid_body = body
            earliest_idx = len(body)
            
            for marker in stop_markers:
                # search for marker
                matches = list(re.finditer(marker, body))
                if matches:
                    idx = matches[0].start()
                    if idx < earliest_idx:
                        earliest_idx = idx
            
            valid_body = body[:earliest_idx]
            
            # Clean up the end (remove trailing empty lines, etc)
            valid_body = clean_latex_block(valid_body)
            
            # Store. If duplicate, keep the longer one
            if tag not in derivations or len(valid_body) > len(derivations[tag]['body']):
                derivations[tag] = {'title': title, 'body': valid_body}

    print(f"Extracted {len(derivations)} unique derivations.")
    
    # 2. Extract Table Rows (The Errors)
    # Pattern: \textbf{N. Title} & Col2 & Col3 \\
    # We look for lines containing & and Ending with \\ (ignoring newlines inside)
    # This is harder with regex.
    # But we can find \textbf{N. ...} blocks.
    
    errors = {} # N -> {title, error, fix}
    
    # Regex for table row:
    # \textbf{1. Title} & ... & ... \\
    # Note: DOTALL is important
    # We iterate over the whole content for this.
    
    row_pattern = r'\\textbf\{(\d+)\.\s+(.*?)\}\s*&\s*(.*?)\s*&\s*(.*?)\s*\\\\'
    found_rows = re.findall(row_pattern, content, flags=re.DOTALL)
    
    # This might fail if rows span multiple lines and regex is greedy.
    # Let's try non-greedy .*?
    
    # A safer way: Split by \hline?
    
    # Let's try to construct the document now.
    
    sorted_tags = sorted(derivations.keys(), key=lambda x: (len(x), x)) # A, B, ..., AA, AB...
    
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(r"""\section{Executive Summary of Mathematical Fixes and Audit}
\label{sec:math_audit_summary}

Based on the provided text, this document is an ``Audit'' and ``Roadmap'' that explicitly identifies mathematical errors, circular arguments, and heuristic gaps found in standard approaches to the Yang-Mills Mass Gap problem. It then proposes specific rigorous ``Derivations'' for each.

\subsection{Index of Mathematical Derivations}
The following is the complete set of 55 Mathematical Derivations (A--BC) required to rigorous prove the Mass Gap.

""")
        
        for tag in sorted_tags:
            d = derivations[tag]
            out.write(f"\\subsubsection{{Derivation {tag}: {d['title']}}}\n")
            out.write(d['body'] + "\n\n")

    print(f"Written to {output_path}")

run()
