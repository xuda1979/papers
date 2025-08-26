from __future__ import annotations
import math, json, platform
import numpy as np
import psutil

def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    margin = (z / denom) * math.sqrt((phat*(1-phat)/n) + (z**2)/(4*n**2))
    return max(0.0, center - margin), min(1.0, center + margin)

def env_info():
    try:
        import numpy, scipy, pandas, matplotlib
        npv = numpy.__version__
        spv = scipy.__version__
        pdv = pandas.__version__
        mtv = matplotlib.__version__
    except Exception:
        npv = spv = pdv = mtv = "unknown"
    info = {
        "python": platform.python_version(),
        "numpy": npv, "scipy": spv, "pandas": pdv, "matplotlib": mtv,
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True)
    }
    return info

def dump_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
