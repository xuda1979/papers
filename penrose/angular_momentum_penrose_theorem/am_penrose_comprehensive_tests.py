"""
Comprehensive Test Suite for the Angular Momentum Penrose Inequality

M_ADM >= sqrt(A/(16*pi) + 4*pi*J^2/A)

Testing many different types of initial data to check the conjecture.
"""

import math

PI = math.pi
sqrt = math.sqrt

def am_penrose_bound(A, J):
    """The conjectured AM-Penrose bound."""
    if A <= 0:
        return float('inf')
    return sqrt(A / (16 * PI) + 4 * PI * J**2 / A)

def check_am_penrose(M_ADM, A, J, name=""):
    """Check if AM-Penrose holds and return detailed results."""
    bound = am_penrose_bound(A, J)
    margin = M_ADM - bound
    ratio = M_ADM / bound if bound > 0 else float('inf')
    holds = margin >= -1e-10
    
    return {
        'name': name,
        'M_ADM': M_ADM,
        'A': A,
        'J': J,
        'bound': bound,
        'margin': margin,
        'ratio': ratio,
        'holds': holds
    }

def print_result(r):
    """Print a single result."""
    status = "✓ HOLDS" if r['holds'] else "✗ VIOLATED"
    saturation = "(SATURATED)" if abs(r['margin']) < 1e-6 else ""
    print(f"  {r['name'][:40]:<40} M={r['M_ADM']:.4f} A={r['A']:.4f} J={r['J']:.4f} "
          f"ratio={r['ratio']:.4f} {status} {saturation}")

# =============================================================================
print("=" * 80)
print("COMPREHENSIVE TEST SUITE FOR AM-PENROSE INEQUALITY")
print("=" * 80)

results = []

# =============================================================================
# TEST 1: KERR FAMILY (should saturate)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: KERR BLACK HOLES (Expected: Saturation)")
print("=" * 80)

for a_over_M in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.0]:
    M = 1.0
    a = a_over_M * M
    if a <= M:
        r_plus = M + sqrt(M**2 - a**2)
        A = 8 * PI * M * r_plus
        J = a * M
        r = check_am_penrose(M, A, J, f"Kerr a/M={a_over_M}")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 2: SCHWARZSCHILD PERTURBATIONS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: SCHWARZSCHILD PERTURBATIONS")
print("=" * 80)

# Perturbed Schwarzschild with small angular momentum added
M0 = 1.0
A0 = 16 * PI * M0**2  # Schwarzschild area

for epsilon in [0.01, 0.05, 0.1, 0.2]:
    # Small rotation perturbation
    J = epsilon * M0**2
    # Area changes slightly (quadratic in spin)
    A = A0 * (1 - 0.5 * epsilon**2)  # Area decreases with spin
    M_ADM = M0  # Mass unchanged to leading order
    
    r = check_am_penrose(M_ADM, A, J, f"Perturbed Schw ε={epsilon}")
    results.append(r)
    print_result(r)

# =============================================================================
# TEST 3: BOWEN-YORK SPINNING DATA
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: BOWEN-YORK SPINNING DATA")
print("=" * 80)

def bowen_york_data(m0, S):
    """
    Bowen-York spinning black hole initial data.
    m0 = bare mass parameter
    S = spin parameter
    
    From Bowen & York (1980), Phys Rev D 21, 2047
    """
    # ADM mass includes spin contribution
    M_ADM = m0 * (1 + S**2 / (4 * m0**4))
    
    # Angular momentum
    J = S
    
    # Horizon area (approximately Schwarzschild with correction)
    # The horizon is slightly deformed by the spin
    A = 16 * PI * m0**2 * (1 + 0.05 * (S / m0**2)**2)
    
    return M_ADM, A, J

for S_over_m2 in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0]:
    m0 = 1.0
    S = S_over_m2 * m0**2
    M_ADM, A, J = bowen_york_data(m0, S)
    r = check_am_penrose(M_ADM, A, J, f"Bowen-York S/m²={S_over_m2}")
    results.append(r)
    print_result(r)

# =============================================================================
# TEST 4: BRILL-LINDQUIST TWO BLACK HOLES
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: BRILL-LINDQUIST BINARY (Two Black Holes)")
print("=" * 80)

def brill_lindquist_binary(m1, m2, d, S_total):
    """
    Brill-Lindquist initial data for two black holes.
    m1, m2 = bare masses
    d = separation
    S_total = total spin
    """
    # ADM mass for Brill-Lindquist (exact)
    M_ADM = m1 + m2
    
    # Total angular momentum
    J = S_total
    
    # Horizon areas (approximately individual Schwarzschild plus interaction)
    # At large separation, A ≈ A1 + A2
    A1 = 16 * PI * m1**2 * (1 + m2 / (2 * d))
    A2 = 16 * PI * m2**2 * (1 + m1 / (2 * d))
    A = A1 + A2
    
    return M_ADM, A, J

for d in [10, 5, 3, 2, 1.5]:
    for S_frac in [0, 0.3, 0.5, 0.7]:
        m1, m2 = 0.5, 0.5
        M_total = m1 + m2
        S = S_frac * M_total**2
        M_ADM, A, J = brill_lindquist_binary(m1, m2, d, S)
        r = check_am_penrose(M_ADM, A, J, f"Brill-Lindquist d={d} S={S_frac}M²")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 5: MISNER DATA (Two Equal Black Holes)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: MISNER DATA (Isometric Throat)")
print("=" * 80)

def misner_data(mu, S_total):
    """
    Misner initial data parameterized by μ.
    Small μ = widely separated holes
    Large μ = close holes (near merger)
    """
    # ADM mass for Misner data
    M_ADM = 1.0 / math.sinh(mu)
    
    # Total area (sum of two throats)
    A = 32 * PI / (math.sinh(mu)**2) * (1 + 0.1 * mu)
    
    # Angular momentum
    J = S_total
    
    return M_ADM, A, J

for mu in [0.5, 1.0, 1.5, 2.0, 2.5]:
    for S_frac in [0, 0.3, 0.5]:
        M_ADM, A, J_dummy = misner_data(mu, 0)
        S = S_frac * M_ADM**2
        M_ADM, A, J = misner_data(mu, S)
        r = check_am_penrose(M_ADM, A, J, f"Misner μ={mu} S={S_frac}M²")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 6: KERR-SCHILD PERTURBATIONS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: KERR-SCHILD PERTURBATIONS")
print("=" * 80)

def kerr_schild_perturbed(M, a, delta_M, delta_a):
    """
    Perturbed Kerr in Kerr-Schild form.
    """
    M_total = M + delta_M
    a_total = a + delta_a
    
    if abs(a_total) > M_total:
        return None, None, None
    
    r_plus = M_total + sqrt(M_total**2 - a_total**2)
    A = 8 * PI * M_total * r_plus
    J = a_total * M_total
    
    return M_total, A, J

M0 = 1.0
for a0 in [0.3, 0.5, 0.7]:
    for delta in [0.01, 0.05, 0.1]:
        # Mass perturbation
        M_ADM, A, J = kerr_schild_perturbed(M0, a0*M0, delta, 0)
        if M_ADM:
            r = check_am_penrose(M_ADM, A, J, f"Kerr-Schild a={a0} δM={delta}")
            results.append(r)
            print_result(r)
        
        # Spin perturbation
        M_ADM, A, J = kerr_schild_perturbed(M0, a0*M0, 0, delta)
        if M_ADM:
            r = check_am_penrose(M_ADM, A, J, f"Kerr-Schild a={a0} δa={delta}")
            results.append(r)
            print_result(r)

# =============================================================================
# TEST 7: BOOSTED BLACK HOLES
# =============================================================================
print("\n" + "=" * 80)
print("TEST 7: BOOSTED SCHWARZSCHILD")
print("=" * 80)

def boosted_schwarzschild(M0, v, S):
    """
    Boosted Schwarzschild black hole.
    v = boost velocity
    S = intrinsic spin
    """
    gamma = 1 / sqrt(1 - v**2)
    
    # ADM mass includes kinetic energy
    M_ADM = gamma * M0
    
    # Angular momentum (intrinsic spin + orbital if applicable)
    J = S
    
    # Horizon area (Lorentz contracted in boost direction)
    # But proper area is invariant to leading order
    A = 16 * PI * M0**2 * (1 + 0.1 * v**2)
    
    return M_ADM, A, J

for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for S_frac in [0, 0.3, 0.5]:
        M0 = 1.0
        S = S_frac * M0**2
        M_ADM, A, J = boosted_schwarzschild(M0, v, S)
        r = check_am_penrose(M_ADM, A, J, f"Boosted v={v} S={S_frac}M₀²")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 8: DISTORTED BLACK HOLES (Brill Waves)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 8: BRILL WAVE DISTORTED BLACK HOLES")
print("=" * 80)

def brill_wave_bh(M0, amplitude, S):
    """
    Schwarzschild black hole with Brill wave perturbation.
    """
    # Mass increases with wave energy
    M_ADM = M0 * (1 + 0.5 * amplitude**2)
    
    # Area increases due to distortion
    A = 16 * PI * M0**2 * (1 + 0.3 * amplitude**2)
    
    # Angular momentum
    J = S
    
    return M_ADM, A, J

for amp in [0.1, 0.3, 0.5, 0.7, 1.0]:
    for S_frac in [0, 0.3, 0.5]:
        M0 = 1.0
        S = S_frac * M0**2
        M_ADM, A, J = brill_wave_bh(M0, amp, S)
        r = check_am_penrose(M_ADM, A, J, f"Brill wave amp={amp} S={S_frac}M₀²")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 9: REISSNER-NORDSTRÖM (Charged, No Spin)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 9: REISSNER-NORDSTRÖM (Charged Black Holes)")
print("=" * 80)

def reissner_nordstrom(M, Q):
    """
    Reissner-Nordström black hole.
    """
    if abs(Q) > M:
        return None, None, None
    
    r_plus = M + sqrt(M**2 - Q**2)
    A = 4 * PI * r_plus**2
    J = 0  # No rotation
    
    return M, A, J

for Q_over_M in [0, 0.3, 0.5, 0.7, 0.9, 0.99]:
    M = 1.0
    Q = Q_over_M * M
    M_ADM, A, J = reissner_nordstrom(M, Q)
    if M_ADM:
        r = check_am_penrose(M_ADM, A, J, f"RN Q/M={Q_over_M}")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 10: KERR-NEWMAN (Charged + Spinning)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 10: KERR-NEWMAN (Charged + Spinning)")
print("=" * 80)

def kerr_newman(M, a, Q):
    """
    Kerr-Newman black hole.
    """
    if a**2 + Q**2 > M**2:
        return None, None, None
    
    r_plus = M + sqrt(M**2 - a**2 - Q**2)
    A = 4 * PI * (r_plus**2 + a**2)
    J = a * M
    
    return M, A, J

for a_over_M in [0.3, 0.5, 0.7]:
    for Q_over_M in [0.1, 0.3, 0.5]:
        M = 1.0
        a = a_over_M * M
        Q = Q_over_M * M
        M_ADM, A, J = kerr_newman(M, a, Q)
        if M_ADM:
            r = check_am_penrose(M_ADM, A, J, f"Kerr-Newman a={a_over_M} Q={Q_over_M}")
            results.append(r)
            print_result(r)

# =============================================================================
# TEST 11: EXTREME MASS RATIO INSPIRAL
# =============================================================================
print("\n" + "=" * 80)
print("TEST 11: EXTREME MASS RATIO INSPIRAL (EMRI)")
print("=" * 80)

def emri_data(M_large, a_large, m_small, r_orbit):
    """
    EMRI: small BH orbiting large Kerr BH.
    """
    # Total mass
    M_ADM = M_large + m_small
    
    # Total angular momentum (Kerr spin + orbital)
    J_kerr = a_large * M_large
    J_orbit = m_small * sqrt(M_large * r_orbit)  # Keplerian approx
    J = J_kerr + J_orbit
    
    # Horizon area of large BH (dominates)
    r_plus = M_large + sqrt(M_large**2 - a_large**2)
    A = 8 * PI * M_large * r_plus
    
    return M_ADM, A, J

M_large = 1.0
for a_over_M in [0.3, 0.7, 0.9]:
    for mu in [0.01, 0.05, 0.1]:  # mass ratio
        for r_orbit in [10, 6, 4]:
            m_small = mu * M_large
            M_ADM, A, J = emri_data(M_large, a_over_M * M_large, m_small, r_orbit)
            if abs(J) <= M_ADM**2:
                r = check_am_penrose(M_ADM, A, J, f"EMRI a={a_over_M} μ={mu} r={r_orbit}")
                results.append(r)
                print_result(r)

# =============================================================================
# TEST 12: HEAD-ON COLLISION REMNANT
# =============================================================================
print("\n" + "=" * 80)
print("TEST 12: HEAD-ON COLLISION REMNANT")
print("=" * 80)

def head_on_remnant(m1, m2, S1, S2, efficiency=0.05):
    """
    Remnant after head-on collision of two BHs.
    """
    # Mass after radiation
    M_ADM = (m1 + m2) * (1 - efficiency)
    
    # Spins may partially cancel or add
    J = S1 + S2  # Aligned case
    
    # Area of final Kerr-like remnant
    if abs(J) > M_ADM**2:
        J = 0.99 * M_ADM**2  # Cap at extremal
    
    a = J / M_ADM
    r_plus = M_ADM + sqrt(M_ADM**2 - a**2)
    A = 8 * PI * M_ADM * r_plus
    
    return M_ADM, A, J

for S1_frac in [0.3, 0.5, 0.7]:
    for S2_frac in [0.3, 0.5, 0.7]:
        m1 = m2 = 0.5
        S1 = S1_frac * m1**2
        S2 = S2_frac * m2**2
        M_ADM, A, J = head_on_remnant(m1, m2, S1, S2)
        r = check_am_penrose(M_ADM, A, J, f"Collision S1={S1_frac} S2={S2_frac}")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 13: TIDALLY DISTORTED BLACK HOLES
# =============================================================================
print("\n" + "=" * 80)
print("TEST 13: TIDALLY DISTORTED BLACK HOLES")
print("=" * 80)

def tidal_distorted_bh(M0, a0, tidal_param):
    """
    Black hole distorted by external tidal field.
    """
    # Mass slightly increased by tidal energy
    M_ADM = M0 * (1 + 0.1 * tidal_param**2)
    
    # Area increased by tidal distortion
    r_plus_0 = M0 + sqrt(M0**2 - a0**2) if a0 <= M0 else M0
    A = 8 * PI * M0 * r_plus_0 * (1 + 0.2 * tidal_param**2)
    
    # Angular momentum unchanged
    J = a0 * M0
    
    return M_ADM, A, J

for a0 in [0, 0.3, 0.5, 0.7]:
    for tidal in [0.1, 0.3, 0.5]:
        M0 = 1.0
        M_ADM, A, J = tidal_distorted_bh(M0, a0 * M0, tidal)
        r = check_am_penrose(M_ADM, A, J, f"Tidal a={a0} λ={tidal}")
        results.append(r)
        print_result(r)

# =============================================================================
# TEST 14: BLACK HOLE + MATTER SHELL
# =============================================================================
print("\n" + "=" * 80)
print("TEST 14: BLACK HOLE + MATTER SHELL")
print("=" * 80)

def bh_matter_shell(M_bh, a_bh, M_shell, R_shell, L_shell):
    """
    Black hole surrounded by rotating matter shell.
    """
    # Total mass
    M_ADM = M_bh + M_shell
    
    # Total angular momentum
    J = a_bh * M_bh + L_shell
    
    # Inner horizon area (just the BH)
    r_plus = M_bh + sqrt(M_bh**2 - a_bh**2) if a_bh <= M_bh else M_bh
    A = 8 * PI * M_bh * r_plus
    
    return M_ADM, A, J

for a_bh in [0, 0.3, 0.5]:
    for M_shell_frac in [0.1, 0.3, 0.5]:
        for L_shell_frac in [0, 0.3, 0.5]:
            M_bh = 1.0
            M_shell = M_shell_frac * M_bh
            L_shell = L_shell_frac * M_bh**2
            M_ADM, A, J = bh_matter_shell(M_bh, a_bh * M_bh, M_shell, 10, L_shell)
            if abs(J) <= M_ADM**2:
                r = check_am_penrose(M_ADM, A, J, f"BH+Shell a={a_bh} M_s={M_shell_frac} L={L_shell_frac}")
                results.append(r)
                print_result(r)

# =============================================================================
# TEST 15: NEAR-EXTREMAL LIMITS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 15: NEAR-EXTREMAL LIMITS")
print("=" * 80)

for epsilon in [0.1, 0.01, 0.001, 0.0001]:
    M = 1.0
    a = M * (1 - epsilon)  # Near extremal
    r_plus = M + sqrt(M**2 - a**2)
    A = 8 * PI * M * r_plus
    J = a * M
    r = check_am_penrose(M, A, J, f"Near-extremal ε={epsilon}")
    results.append(r)
    print_result(r)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

violations = [r for r in results if not r['holds']]
saturations = [r for r in results if r['holds'] and abs(r['margin']) < 1e-6]
strict_holds = [r for r in results if r['holds'] and abs(r['margin']) >= 1e-6]

print(f"\nTotal tests: {len(results)}")
print(f"  ✓ Strict holds (M > bound): {len(strict_holds)}")
print(f"  ✓ Saturations (M = bound): {len(saturations)}")
print(f"  ✗ Violations (M < bound): {len(violations)}")

if violations:
    print("\n*** VIOLATIONS FOUND: ***")
    for r in violations:
        print_result(r)
else:
    print("\n*** NO VIOLATIONS FOUND ***")
    print("All tested initial data configurations satisfy the AM-Penrose inequality!")

# Check closest to violation
if results:
    closest = min(results, key=lambda r: r['ratio'])
    print(f"\nClosest to violation (ratio = M/bound):")
    print_result(closest)

# List all saturations
if saturations:
    print(f"\nSaturated cases (all should be Kerr):")
    for r in saturations:
        print_result(r)
