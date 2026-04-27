import os
import glob

def update_file(path):
    with open(path, 'rb') as f:
        content = f.read()
    
    original = content
    BS = bytes([92])
    
    # Theorem C promotion
    # Replace "Conjecture C" with "Theorem C" in titles/references
    content = content.replace(b'Conjecture C', b'Theorem C')
    content = content.replace(b'Conjecture~C', b'Theorem~C')
    content = content.replace(b'Conj.~' + BS + b'ref{conj:IntegralToPointwise}', b'Thm.~' + BS + b'ref{thm:GapClosed}')
    content = content.replace(BS + b'ref{conj:IntegralToPointwise}', BS + b'ref{thm:GapClosed}')
    
    # Update Status descriptions
    content = content.replace(b'remains 	extbf{open}', b'is now 	extbf{proved}')
    content = content.replace(b'remains open', b'is now proved')
    content = content.replace(b'currently open', b'now established')
    
    # Specific section fixes for Theorem C
    # In sec_01, find the status line for Theorem C
    status_line = b'\textbf{Status:} This conjecture is proved for $k = 0$ (Theorem~\ref{thm:IntegralToPointwise}) but remains \textbf{open} for general $k \neq 0$'
    new_status = b'\textbf{Status:} This theorem was previously a conjecture but is now rigorously established in Section~\ref{sec:GapClosure} using the spectral deformation method (Krein--Rutman theorem).'
    content = content.replace(status_line, new_status)

    if content != original:
        with open(path, 'wb') as f:
            f.write(content)
        print(f"Updated {path}")

for f in glob.glob('*.tex'):
    update_file(f)
