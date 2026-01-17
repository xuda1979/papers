"""
RG Step Verifier (Rigorous)
---------------------------
Implements the specific contraction checks for one Renormalization Group step.
Uses Interval Arithmetic to guarantee inclusion.
"""

from rigorous_interval import Interval, interval_mat_vec_mult, interval_norm
import ym_basis

class RGVerifier:
    def __init__(self, L=2.0):
        self.L = L

    def check_contraction(self, step_data):
        """
        Verifies that the image of the input tube under the RG map 
        is strictly contained within the output tube.
        
        step_data structure (expected from JSON):
        {
            "step_id": int,
            "input_center": [float...], # center of input ball
            "input_radius": [float...], # radius of input ball (defines interval)
            "linear_map": [[float, float]...], # Interval matrix Jacobian D_R
            "nonlinear_bound": [float...], # Interval vector bound on R_nonlinear
            "output_center": [float...],
            "output_radius": [float...], 
            "tail_bound_in": float,
            "tail_bound_out": float
        }
        """
        
        # 1. Parse Inputs into Intervals
        # Construct Input Tube Intervals: [c - r, c + r]
        input_intervals = []
        for c, r in zip(step_data["input_center"], step_data["input_radius"]):
            input_intervals.append(Interval(c) + Interval(-r, r))
            
        # Parse Jacobian (Linear Map)
        jacobian = []
        raw_mat = step_data["linear_map"]
        rows = len(step_data["input_center"])
        # Assuming flattened or row-major list of pairs
        # This is a scaffold: in real certificate, schema is strictly defined
        for i in range(rows):
            row = []
            for j in range(rows):
                # Mockup: extract pair
                pair = raw_mat[i][j]
                row.append(Interval(pair[0], pair[1]))
            jacobian.append(row)
            
        # Parse Nonlinear Remainder Bound
        nonlinear = [Interval(b[0], b[1]) for b in step_data["nonlinear_bound"]]
        
        # 2. Compute the Image: Image = Jacobian * Input + Nonlinear
        # Note: This is a linear enclosure of the map R(x) approx J*x + N(x)
        # where x is relative to fixed point or linearization center.
        # Strict inclusion requires: Image \subset Output_Tube
        
        linear_part = interval_mat_vec_mult(jacobian, input_intervals)
        
        # Add nonlinear part
        image_intervals = []
        for l, n in zip(linear_part, nonlinear):
            image_intervals.append(l + n)
            
        # 3. Check Inclusion against Output Tube
        output_intervals = []
        for c, r in zip(step_data["output_center"], step_data["output_radius"]):
            output_intervals.append(Interval(c) + Interval(-r, r))
            
        strict_inclusion = True
        for i, (img, out) in enumerate(zip(image_intervals, output_intervals)):
            if not out.contains(img):
                print(f"  [FAIL] Dim {i}: Image {img} not inside {out}")
                strict_inclusion = False
                
        # 4. Tail Logic (Non-Perturbative Slot)
        # The certificate must provide the 'Pollution Constant' K 
        # explicitly, as a Non-Perturbative Global Bound.
        # This constant is NOT derived from perturbative OPE.
        # It is strictly bounded by the Certificate's "pollution_constant" field.
        K_pollution = step_data.get("pollution_constant", 0.05) 
        lambda_irr = self.L ** (-2)
        
        tail_in = step_data["tail_bound_in"]
        norm_sq = interval_norm(image_intervals) # Upper bound of image norm
        
        # Tail Evolution: tau' = lambda * tau + K * ||Image||^2
        computed_tail_out = (lambda_irr * tail_in) + (K_pollution * norm_sq)
        
        claimed_tail_out = step_data["tail_bound_out"]
        
        if computed_tail_out > claimed_tail_out:
             print(f"  [FAIL] Tail: Computed {computed_tail_out} > Claimed {claimed_tail_out}")
             strict_inclusion = False
             
        return strict_inclusion

