import os
import re
import glob

def find_latex_errors(root_dir):
    tex_files = glob.glob(os.path.join(root_dir, "**/*.tex"), recursive=True)
    
    labels = set()
    defined_labels = {}
    references = []
    
    errors = []

    # Regex for labels and refs
    label_re = re.compile(r'\\label\{([^}]+)\}')
    ref_re = re.compile(r'\\ref\{([^}]+)\}')
    cite_re = re.compile(r'\\cite\{([^}]+)\}')
    
    # scan for definitions first
    for file_path in tex_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
            continue
            
        for i, line in enumerate(lines):
            # Check for labels
            for match in label_re.finditer(line):
                lbl = match.group(1)
                if lbl in defined_labels:
                    errors.append(f"Duplicate label '{lbl}' in {os.path.basename(file_path)}:{i+1}. First defined in {os.path.basename(defined_labels[lbl]['file'])}:{defined_labels[lbl]['line']}")
                else:
                    defined_labels[lbl] = {'file': file_path, 'line': i+1}
                    labels.add(lbl)

            # Collect refs
            for match in ref_re.finditer(line):
                references.append({'lbl': match.group(1), 'file': file_path, 'line': i+1, 'type': 'ref'})
            
            # Check for basic unbalanced braces per line (heuristic, multiline is hard)
            # This is naive but catches simple typos
            open_braces = line.count('{')
            close_braces = line.count('}')
            if open_braces != close_braces:
                # exclude comments
                if '%' in line:
                    if line.find('%') < line.find('{') or line.find('%') < line.find('}'):
                         pass # naive check, ignore lines with comments for now to avoid false positives
                    else:
                         if line.split('%')[0].count('{') != line.split('%')[0].count('}'):
                              errors.append(f"Possible unbalanced braces in {os.path.basename(file_path)}:{i+1} : {line.strip()}")
                else:
                     errors.append(f"Possible unbalanced braces in {os.path.basename(file_path)}:{i+1} : {line.strip()}")
            
            # Check for common math errors
            if "$" in line:
                # count $ - if odd, likely error (unless multiline)
                # Filter escaped \$
                cnt = line.count("$") - 2 * line.count("\\$")
                if cnt % 2 != 0:
                     # This is common in valid latex if math spans lines, but worth listing as potential
                     pass # Too many false positives for multiline
            
            # Check for specific suspicious patterns
            if "\\begin{equation}" in line and "\\end{equation}" not in line:
                 pass # multiline, ignore
            
            # Typos
            if "\\begin" in line: # fixed typo check
                pass
            if "\\frac" in line and "}{" not in line:
                 # \frac{a}{b} must have }{. If not, might be \frac{a} {b} (valid) or \frac ab (valid for single chars).
                 # But \frac{a}{b} is standard. 
                 pass

    # Check references
    for ref in references:
        if ref['lbl'] not in labels:
            # Check if it's a standard external label or just missing
            errors.append(f"Undefined reference '{ref['lbl']}' in {os.path.basename(ref['file'])}:{ref['line']}")

    return errors

if __name__ == "__main__":
    root = r"c:\Users\Lenovo\papers\penrose\angular_momentum_penrose_theorem\split"
    errs = find_latex_errors(root)
    if errs:
        print("Found errors:")
        for e in errs:
            print(e)
    else:
        print("No obvious errors found.")
