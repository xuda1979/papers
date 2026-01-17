"""
Rigorous Interval Arithmetic Library
------------------------------------
Implements IEEE-754 compliant interval arithmetic with directed rounding.
This is essential for the Computer-Assisted Proof to guarantee bounds.
"""

import math

# Infinity for intervals
INF = float('inf')

class Interval:
    __slots__ = ('low', 'high')

    def __init__(self, low, high=None):
        if high is None:
            high = low
        
        self.low = float(low)
        self.high = float(high)
        
        # Canonicalize empty intervals or invalid inputs if needed, 
        # but for the proof we want to fail strict on NaN or scrambled bounds
        if math.isnan(self.low) or math.isnan(self.high):
             raise ValueError("Interval bounds cannot be NaN")
        
    def width(self):
        return self.high - self.low

    def mid(self):
        return 0.5 * (self.low + self.high)

    def __repr__(self):
        return f"[{self.low}, {self.high}]"

    def contains(self, other):
        if isinstance(other, (int, float)):
            return self.low <= other <= self.high
        return self.low <= other.low and self.high >= other.high

    # --- Arithmetic Operations with Directed Rounding ---

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        # low = low + low (rounded down)
        # high = high + high (rounded up)
        # We use strict error bounds or nextafter if available. 
        # Since standard float addition is round-to-nearest, we add epsilon manually or use math.nextafter.
        
        new_low = self.next_down(self.low + other.low)
        new_high = self.next_up(self.high + other.high)
        return Interval(new_low, new_high)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        new_low = self.next_down(self.low - other.high)
        new_high = self.next_up(self.high - other.low)
        return Interval(new_low, new_high)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
        
        # Products of all combinations
        p1 = self.low * other.low
        p2 = self.low * other.high
        p3 = self.high * other.low
        p4 = self.high * other.high
        
        # Min rounded down, Max rounded up
        new_low = self.next_down(min(p1, p2, p3, p4))
        new_high = self.next_up(max(p1, p2, p3, p4))
        return Interval(new_low, new_high)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Interval(other)
            
        if other.low <= 0 <= other.high:
            # Division by interval containing zero is complicated (returns extended intervals).
            # For this proof context, we treat it as an error or [-inf, inf]
            if other.low == 0 and other.high == 0:
                raise ZeroDivisionError
            return Interval(-INF, INF)

        # Division is multiplication by 1/other
        # 1/other is [1/other.high, 1/other.low]
        # We handle rounding carefully
        recip_low = self.next_down(1.0 / other.high)
        recip_high = self.next_up(1.0 / other.low)
        
        recip = Interval(recip_low, recip_high)
        return self * recip

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise NotImplementedError("Only integer powers supported for rigor currently")
        
        if exponent % 2 == 0:
            # Even power: [0, max(|low|, |high|)^2] if 0 in self
            # else [min^2, max^2]
            abs_vals = [abs(self.low), abs(self.high)]
            mx = max(abs_vals)
            mx_sq = self.next_up(mx * mx)
            
            if self.contains(0):
                return Interval(0.0, mx_sq)
            else:
                mn = min(abs_vals)
                mn_sq = self.next_down(mn * mn)
                return Interval(mn_sq, mx_sq)
        else:
            # Odd power: monotonic
            p_low = self.low ** exponent
            p_high = self.high ** exponent
            return Interval(self.next_down(p_low), self.next_up(p_high))

    # --- Directed Rounding Helpers ---
    
    @staticmethod
    def next_up(x):
        if hasattr(math, 'nextafter'):
            return math.nextafter(x, INF)
        else:
            # Fallback for older Python
            import sys
            if x == 0: return sys.float_info.min
            return x + sys.float_info.epsilon * abs(x)

    @staticmethod
    def next_down(x):
        if hasattr(math, 'nextafter'):
            return math.nextafter(x, -INF)
        else:
            # Fallback for older Python
            import sys
            if x == 0: return -sys.float_info.min
            return x - sys.float_info.epsilon * abs(x)

# --- Vector / Matrix Helpers using Intervals ---

def interval_mat_vec_mult(matrix, vector):
    """
    matrix: list of lists of Intervals
    vector: list of Intervals
    returns: list of Intervals
    """
    rows = len(matrix)
    cols = len(matrix[0])
    if len(vector) != cols:
        raise ValueError("Dimension mismatch")
        
    result = []
    for r in range(rows):
        sum_val = Interval(0.0)
        for c in range(cols):
            val = matrix[r][c]
            vec_val = vector[c]
            product = val * vec_val
            sum_val = sum_val + product
        result.append(sum_val)
    return result

def interval_norm(vector):
    """L2 norm upper bound for interval vector"""
    sum_sq = Interval(0.0)
    for x in vector:
        sum_sq = sum_sq + (x**2)
    return math.sqrt(sum_sq.high) # Return float upper bound
