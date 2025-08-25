import numpy as np
from simulation import logical_error_rate

def test_error_rate_monotone():
    # with more physical error, logical error should (weakly) increase on average
    d = 5
    sigma = 0.25
    shots = 2000
    L1 = logical_error_rate(d, 1e-4, shots, sigma, "analog", 0.0, "bit_flip")
    L2 = logical_error_rate(d, 1e-2, shots, sigma, "analog", 0.0, "bit_flip")
    assert L2 >= L1

def test_decoder_switches_behavior():
    d = 3
    sigma = 0.25
    shots = 100000
    L_analog = logical_error_rate(d, 5e-2, shots, sigma, "analog", 0.0, "bit_flip")
    L_digital = logical_error_rate(d, 5e-2, shots, sigma, "digital", 0.0, "bit_flip")
    assert L_analog != L_digital
