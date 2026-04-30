content = r"""% Minimal honest master file used by repository audits.
\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
Xbegin{document}
\input{app128_roadmap4_enhanced}
\input{app151_hamiltonian_identification_theorem_boundary}
\input{app174_holographic_stochastic_transport}
\end{document}
""".replace('Xbegin', '\begin')

with open("/Users/daxu/papers/yang/yang-mills/split/yang_mills_master.tex", "w") as f:
    f.write(content)
