import base64

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/b64.txt', 'r') as f:
    b64_str = f.read().strip()

new_math = base64.b64decode(b64_str).decode('utf-8')

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_34_logical_structure_and_gap_closure.tex', 'r') as f:
    text = f.read()

start_marker = r"\subsection{Adjoint Calibration and the Pointwise Resolution of Conjecture C}"
end_marker = r"\subsection{Disjointness of Singular Sets}"

start_idx = text.find(start_marker)
if start_idx == -1:
    # If the previous bad patch was already reverted:
    start_marker = r"\subsection{Adjoint-Cone Replacement for the Former Gap}"
    start_idx = text.find(start_marker)

end_idx = text.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Markers not found!")
    exit(1)

new_text = text[:start_idx] + new_math + text[end_idx:]

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_34_logical_structure_and_gap_closure.tex', 'w') as f:
    f.write(new_text)

print("Replacement successful.")
