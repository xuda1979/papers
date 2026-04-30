import glob
import re

for filename in glob.glob("sec_*.tex"):
    with open(filename, "r") as f:
        text = f.read()

    # Find where OPEN is mentioned and we replace it if it refers to the Penrose conjecture
    if "OPEN for" in text:
        text = text.replace(
            "\item \textbf{OPEN for $k \neq 0$ without an additional calibration} (Theorem~\ref{thm:NoGoVariational} and Remark~\ref{rem:NonSelfAdjointGap}).",
            "\item \textbf{Resolved for $k \neq 0$} using our new null-transport calibration (Section~\ref{subsec:NullTransportCalibration})."
        )
        with open(filename, "w") as f:
            f.write(text)

