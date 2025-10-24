import time, threading
from typing import Optional
try:
    import pynvml
    NVML_OK = True
except Exception:
    NVML_OK = False
import subprocess

class EnergyMeter:
    """
    Best-effort GPU energy measurement.
    - If NVML energy counters exist, use them.
    - Else sample power via NVML or `nvidia-smi --query-gpu=power.draw`.
    """
    def __init__(self, enable:bool=True, device_index:int=0, sample_hz:float=10.0):
        self.enable = enable and NVML_OK
        self.dev = None
        self.device_index = device_index
        self.sample_hz = sample_hz
        self._thread=None
        self._stop=False
        self._energy_j=0.0
        self._prev_mJ=None

    def session(self):
        return _EnergySession(self)

    def _start(self):
        if not self.enable:
            return
        pynvml.nvmlInit()
        self.dev = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        # Try total energy
        try:
            self._prev_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.dev)
        except Exception:
            self._prev_mJ = None
        self._stop=False
        self._energy_j = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        dt = 1.0/max(1e-6, self.sample_hz)
        while not self._stop:
            try:
                if self._prev_mJ is not None:
                    mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.dev)
                    self._energy_j += max(0.0, (mJ - self._prev_mJ))/1000.0
                    self._prev_mJ = mJ
                else:
                    p_mw = pynvml.nvmlDeviceGetPowerUsage(self.dev)  # milliwatts
                    self._energy_j += (p_mw/1000.0) * dt
            except Exception:
                pass
            time.sleep(dt)

    def _stop_session(self):
        if not self.enable: return
        self._stop=True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def last_step_energy_j(self)->float:
        return float(self._energy_j)

class _EnergySession:
    def __init__(self, meter:EnergyMeter):
        self.meter = meter
    def __enter__(self):
        self.meter._start()
        return self.meter
    def __exit__(self, exc_type, exc, tb):
        self.meter._stop_session()
