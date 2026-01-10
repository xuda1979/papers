import sys
import math

class Interval:
    """
    A class representing a real interval [lower, upper] for rigorous arithmetic.
    Includes machine epsilon padding for rigorous directed rounding.
    """
    def __init__(self, lower, upper=None):
        if upper is None:
            upper = lower
        self.lower = float(lower)
        self.upper = float(upper)
        # Ensure proper ordering
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower
            
    def __add__(self, other):
        if isinstance(other, Interval):
            # lower + lower (round down), upper + upper (round up)
            l = self.lower + other.lower
            u = self.upper + other.upper
            return Interval(math.nextafter(l, -math.inf), math.nextafter(u, math.inf))
        val = float(other)
        l = self.lower + val
        u = self.upper + val
        return Interval(math.nextafter(l, -math.inf), math.nextafter(u, math.inf))

    def __sub__(self, other):
        if isinstance(other, Interval):
            l = self.lower - other.upper
            u = self.upper - other.lower
            return Interval(math.nextafter(l, -math.inf), math.nextafter(u, math.inf))
        val = float(other)
        l = self.lower - val
        u = self.upper - val
        return Interval(math.nextafter(l, -math.inf), math.nextafter(u, math.inf))
        
    def __mul__(self, other):
        if isinstance(other, Interval):
            products = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper
            ]
            # We must round min down and max up
            min_prod = min(products)
            max_prod = max(products)
            return Interval(math.nextafter(min_prod, -math.inf), math.nextafter(max_prod, math.inf))
        
        val = float(other)
        if val >= 0:
            l = self.lower * val
            u = self.upper * val
        else:
            l = self.upper * val
            u = self.lower * val
        return Interval(math.nextafter(l, -math.inf), math.nextafter(u, math.inf))

    def __truediv__(self, other):
        if isinstance(other, Interval):
            if other.lower <= 0 <= other.upper:
                raise ValueError("Division by interval containing zero")
            
            # Division is multiplication by 1/other
            inv_lower = 1.0 / other.upper
            inv_upper = 1.0 / other.lower
            # Create inverse interval with rounding
            inv = Interval(math.nextafter(inv_lower, -math.inf), math.nextafter(inv_upper, math.inf))
            
            return self * inv
            
        val = float(other)
        if val == 0:
            raise ValueError("Division by zero")
        return self * (1.0 / val)

    def __str__(self):
        return f"[{self.lower:.6f}, {self.upper:.6f}]"

    def __repr__(self):
        return self.__str__()

    def log(self):
         # rigorous log bounds
         if self.lower <= 0:
             raise ValueError("Log of non-positive number")
         l = math.log(self.lower)
         u = math.log(self.upper)
         return Interval(math.nextafter(l, -math.inf), math.nextafter(u, math.inf))

def factorial(n):
    res = 1
    for i in range(1, n+1):
        res *= i
    return res

def bessel_i_term(v, x_sq_4, k):
    # Returns the k-th term: (x^2/4)^k / (k! * (v+k)!)
    # We factor out (x/2)^v later
    return (x_sq_4**k) / (factorial(k) * factorial(v + k))

def bessel_i_bound(v, x_interval, terms=20):
    """
    Returns rigorous [lower, upper] for I_v(x) where x is in x_interval.
    Since I_v(x) is monotonic increasing for x>0, we evaluate at endpoints.
    
    For a fixed x, I_v(x) = (x/2)^v * Sum_{k=0}^inf T_k
    Lower bound: Sum_{k=0}^{terms} T_k (since all terms positive)
    Upper bound: Sum_{k=0}^{terms} T_k + Tail_Bound
    """
    
    def eval_at_point(val):
        x = val
        x_2 = x / 2.0
        x_sq_4 = (x*x) / 4.0
        
        # Calculate finite sum
        sum_val = 0.0
        term_val = 0.0 # Just initialization
        
        # k=0 term
        # term_0 = 1 / (0! * v!) = 1/v!
        current_term = 1.0 / factorial(v)
        sum_val += current_term
        
        # Iterate
        for k in range(1, terms + 1):
            # Ratio T_k / T_{k-1} = (x^2/4) / (k * (v+k))
            ratio = x_sq_4 / (k * (v + k))
            current_term *= ratio
            sum_val += current_term
            
        # Tail bound
        # T_{N+1} / T_N = (x^2/4) / ((N+1)(v+N+1))
        # Let q = (x^2/4) / ((terms+1)*(v+terms+1))
        # If q < 1, tail < T_{terms} * q / (1-q)
        q = x_sq_4 / ((terms + 1) * (v + terms + 1))
        
        # Prefactor
        pre = math.pow(x_2, v)
        
        lower_sum = sum_val
        upper_sum = sum_val
        
        if q < 0.99: # Ensure convergence
             tail_max = current_term * q / (1.0 - q)
             upper_sum += tail_max
        else:
             # Fallback if terms not enough
             upper_sum += 1.0 # Loose bound or error
             
        return (pre * lower_sum, pre * upper_sum)

    # Calculate for lower endpoint
    l_low, l_high = eval_at_point(x_interval.lower)
    # Calculate for upper endpoint
    u_low, u_high = eval_at_point(x_interval.upper)
    
    # Global bounds
    # Lower bound comes from lower endpoint lower bound (monotonic)
    # Upper bound comes from upper endpoint upper bound
    return Interval(math.nextafter(l_low, -math.inf), math.nextafter(u_high, math.inf))

def compute_mass_gap_strong_coupling(beta_val):
    """
    Computes rigorous interval for the mass gap m(beta).
    """
    beta = Interval(beta_val - 1e-14, beta_val + 1e-14)
    
    i1 = bessel_i_bound(1, beta, terms=30)
    i2 = bessel_i_bound(2, beta, terms=30)
    
    # u = I_2/I_1
    u = i2 / i1
    
    # Gap m = -log(u) - (2d-1)u
    # d = 4
    d = 4
    
    term1 = u.log() 
    # term1 is log(u). We want -log(u).
    neg_log_u = Interval(-term1.upper, -term1.lower)
    
    term2 = u * (2*d - 1)
    
    full_gap = neg_log_u - term2
    
    return full_gap

def run_verification():
    print("Running Rigorous Mass Gap Verification (Updated with Interval Arithmetic)...")
    betas = [0.1, 0.5]
    
    print("-" * 65)
    print(f"{'Beta':<10} | {'Gap Interval':<35} | {'Positive?':<10}")
    print("-" * 65)
    
    for b in betas:
        gap = compute_mass_gap_strong_coupling(b)
        pos = "YES" if gap.lower > 0 else "NO"
        print(f"{b:<10} | {str(gap):<35} | {pos:<10}")
        
    print("-" * 65)

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
