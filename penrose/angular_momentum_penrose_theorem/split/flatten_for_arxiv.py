"""
Script to flatten the Angular Momentum Penrose Inequality paper
into a single tex file for arXiv submission.
"""
import os
import re

# Define the order of files as specified in main.tex
file_order = [
    "preamble_clean.tex",
    "title-abstract.tex",
    "sec01-introduction.tex",
    "sec02-kerr.tex",
    "sec03-proof-outline.tex",
    "sec04-jang.tex",
    "sec05-lichnerowicz.tex",
    "sec06-amo.tex",
    "sec07-subextremality.tex",
    "sec08-synthesis.tex",
    "sec09-rigidity.tex",
    "sec10-extensions.tex",
    "sec13-conclusion.tex",
    # appendix files
    "sec11-numerical.tex",
    "sec12-technical.tex",
    "app01-amo-estimates.tex",
    "app02-schauder.tex",
    "app03-supersolution.tex",
    "app04-subext-improvement.tex",
    "app05-mars-simon.tex",
    "app06-function-spaces.tex",
    "acknowledgments.tex",
    "bibliography.tex",
]

def read_file(filename):
    """Read a file and return its contents."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            return f.read()

def create_flattened_file():
    """Create the flattened tex file."""
    output = []
    
    # Document class
    output.append(r"% Single-file version for arXiv submission")
    output.append(r"% Angular Momentum Penrose Inequality paper")
    output.append(r"% Author: Da Xu")
    output.append(r"")
    output.append(r"\documentclass[12pt]{article}")
    output.append(r"")
    
    # Add preamble content (without documentclass if present)
    preamble = read_file("preamble_clean.tex")
    # Remove any documentclass if present
    preamble = re.sub(r'\\documentclass.*?\n', '', preamble)
    output.append(r"% ==================== PREAMBLE ====================")
    output.append(preamble.strip())
    output.append(r"")
    
    # Begin document
    output.append(r"\begin{document}")
    output.append(r"")
    
    # Add title-abstract
    output.append(r"% ==================== TITLE AND ABSTRACT ====================")
    output.append(read_file("title-abstract.tex").strip())
    output.append(r"")
    
    # Add main sections
    main_sections = [
        ("sec01-introduction.tex", "SECTION 1: INTRODUCTION"),
        ("sec02-kerr.tex", "SECTION 2: KERR VERIFICATION"),
        ("sec03-proof-outline.tex", "SECTION 3: PROOF OUTLINE"),
        ("sec04-jang.tex", "SECTION 4: JANG EQUATION"),
        ("sec05-lichnerowicz.tex", "SECTION 5: LICHNEROWICZ EQUATION"),
        ("sec06-amo.tex", "SECTION 6: AMO FLOW"),
        ("sec07-subextremality.tex", "SECTION 7: SUBEXTREMALITY"),
        ("sec08-synthesis.tex", "SECTION 8: SYNTHESIS"),
        ("sec09-rigidity.tex", "SECTION 9: RIGIDITY"),
        ("sec10-extensions.tex", "SECTION 10: EXTENSIONS"),
        ("sec13-conclusion.tex", "SECTION 11: CONCLUSION"),
    ]
    
    for filename, header in main_sections:
        output.append(f"% ==================== {header} ====================")
        content = read_file(filename).strip()
        output.append(content)
        output.append(r"")
    
    # Appendix declaration
    output.append(r"% ==================== APPENDICES ====================")
    output.append(r"\appendix")
    output.append(r"")
    
    # Add appendix sections
    appendix_sections = [
        ("sec11-numerical.tex", "APPENDIX A: NUMERICAL ILLUSTRATIONS"),
        ("sec12-technical.tex", "APPENDIX B: TECHNICAL FOUNDATIONS"),
        ("app01-amo-estimates.tex", "APPENDIX C: AMO ESTIMATES"),
        ("app02-schauder.tex", "APPENDIX D: SCHAUDER ESTIMATES"),
        ("app03-supersolution.tex", "APPENDIX E: SUPERSOLUTION ANALYSIS"),
        ("app04-subext-improvement.tex", "APPENDIX F: SUBEXTREMALITY IMPROVEMENT"),
        ("app05-mars-simon.tex", "APPENDIX G: MARS-SIMON TENSOR"),
        ("app06-function-spaces.tex", "APPENDIX H: FUNCTION SPACES"),
    ]
    
    for filename, header in appendix_sections:
        output.append(f"% ==================== {header} ====================")
        content = read_file(filename).strip()
        output.append(content)
        output.append(r"")
    
    # Add acknowledgments
    output.append(r"% ==================== ACKNOWLEDGMENTS ====================")
    output.append(read_file("acknowledgments.tex").strip())
    output.append(r"")
    
    # Add bibliography
    output.append(r"% ==================== BIBLIOGRAPHY ====================")
    output.append(read_file("bibliography.tex").strip())
    output.append(r"")
    
    # End document
    output.append(r"\end{document}")
    
    return '\n'.join(output)

if __name__ == "__main__":
    flattened = create_flattened_file()
    
    # Write to arxiv_submission.tex
    with open("arxiv_submission.tex", 'w', encoding='utf-8') as f:
        f.write(flattened)
    
    print("Created arxiv_submission.tex")
    print(f"Total length: {len(flattened)} characters")
    print(f"Total lines: {flattened.count(chr(10)) + 1}")
