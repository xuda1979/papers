with open("/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex", "r") as f:
    text = f.read()

import re

text = re.sub(r"\item \textbf\{OPEN for \$k \neq 0\$ without an additional calibration\}.*?\.", r"\item \textbf{Resolved for $k \neq 0$} using our new null-transport calibration (Section~\ref{subsec:NullTransportCalibration}).", text, flags=re.DOTALL)

with open("/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex", "w") as f:
    f.write(text)

