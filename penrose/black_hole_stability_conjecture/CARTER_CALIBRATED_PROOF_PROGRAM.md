# Carter-Calibrated Microlocal Energy: A New Proof Program

## Status

This is an invented mathematical framework aimed at the full-subextremal nonlinear Kerr stability conjecture. It is not an unconditional proof. Its purpose is to replace vague phrases like "hidden symmetry controls superradiance" with a precise new object whose verification would imply the conjecture.

The proposed new math is a **Carter-calibrated positive-commutator calculus** on the trapped-superradiant phase space.

## 1. The Obstruction

For subextremal Kerr, the hardest exterior linear problem is the simultaneous presence of:

- trapping near the Kerr photon region,
- superradiant horizon flux in the frequency range $\omega(\omega-m\Omega_H)<0$,
- derivative loss in transformations between Teukolsky variables and metric perturbations,
- near-extremal degeneration as $|a|\to M$.

Existing vector-field energies handle these effects region by region. What is missing for a full-subextremal nonlinear proof is a coercive energy that stays positive after the trapped-superradiant modes are put back into the nonlinear gauge hierarchy.

## 2. New Object: Carter-Calibrated Symbol

Let $p$ be the principal Hamiltonian of null geodesic flow on Kerr and let

$$
E=-\xi_t,\qquad L=\xi_\phi,\qquad Q=\text{Carter constant}.
$$

On the characteristic set $\{p=0\}$ define the trapped-superradiant set

$$
K_{\mathrm{tsr}}
=\{p=0,\ H_p r=0,\ H_p^2 r=0,\ \omega(\omega-m\Omega_H)<0\}.
$$

A **Carter-calibrated symbol** is a real order-zero symbol

$$
b=b(r,\theta,\xi;M,a)
$$

of the form

$$
b=\beta_0(r,\theta,E,L,Q)
  + H_p g(r,\theta,\xi)
  + \beta_H(r,\theta,\xi)(E-\Omega_H L)^2,
$$

where:

- $\beta_0$ is an algebraic expression in the Kerr separation constants,
- $g$ is an escape function allowed to be singular only at controlled radial turning strata,
- $\beta_H\ge 0$ is supported in the red-shift region,
- the full symbol is invariant under the stationary and axial Killing flows.

The novelty is that $b$ is calibrated against **orbit averages of the Carter invariant**, not only against radial escape. This is meant to convert the trapped-superradiant problem into a finite collection of inequalities on the integrable torus fibration.

## 3. Central New Lemma

**Carter Calibration Lemma.**  
For each fixed subextremal Kerr parameter $|a|<M$, there exist constants $c_\chi>0$, $C_\chi<\infty$ and a Carter-calibrated symbol $b$ such that, on a conic neighborhood of $K_{\mathrm{tsr}}$,

$$
H_p b
+ C_\chi^{-1}(E-\Omega_H L)^2
+ C_\chi^{-1}\operatorname{dist}(\rho,K_{\mathrm{tsr}})^2
\ge c_\chi q_{\mathrm{grav}}(\rho),
$$

where $q_{\mathrm{grav}}$ is the principal positive density controlling the spin-$\pm2$ Teukolsky variables after the Chandrasekhar transformation.

Moreover, $b$ admits a quantization $B=\operatorname{Op}(b)$ whose commutator with the transformed Teukolsky operator satisfies

$$
\operatorname{Re}\langle i[P,B]u,u\rangle
\mathcal F_{\mathcal H^+}[u]
\mathcal F_{\mathcal I^+}[u]
\ge c_\chi\|u\|_{LE^1_{\mathrm{tsr}}}^2
  - C_\chi\|u\|_{LE^0_{\mathrm{lower}}}^2.
$$

This is the new mathematical heart. If true, it removes the final trapped-superradiant sign obstruction at the linear level.

## 4. Why This Might Be Proveable

The null geodesic flow on Kerr is completely integrable. On each trapped torus, any smooth symbol has an orbit average. Therefore one can try to solve the cohomological equation

$$
H_p g = q_{\mathrm{grav}}-\langle q_{\mathrm{grav}}\rangle_{\mathbb T^2}
$$

away from resonances. The resonant tori are lower-dimensional and can be treated by a finite rational certificate:

$$
\langle q_{\mathrm{grav}}\rangle_{\gamma}
\lambda_\gamma(E-\Omega_H L)^2
\mu_\gamma Q
\nu_\gamma(E^2+L^2)
\ge c_\chi
$$

for every resonant closed orbit $\gamma$ in the trapped-superradiant set.

Thus the infinite-dimensional positivity problem is reduced to:

1. a cohomological equation on nonresonant tori,
2. finitely many rational inequalities on resonant strata,
3. a sharp Gårding/Fefferman-Phong quantization step,
4. red-shift absorption at the horizon,
5. Chandrasekhar inverse estimates with controlled derivative loss.

## 5. Nonlinear Use

Assume the Carter Calibration Lemma and the already-known full-subextremal linear stability package can be integrated into a global modulation gauge. Define the nonlinear energy

$$
\mathcal E_s[h]
=\sum_{|\alpha|\le s}
\left(
E_{\mathrm{red}}[\nabla^\alpha h]
+E_{\mathrm{ext}}[\nabla^\alpha h]
+\langle B\nabla^\alpha\Psi[h],\nabla^\alpha\Psi[h]\rangle
\right)
+\mathcal M_s[h],
$$

where $\Psi[h]$ denotes the curvature/Teukolsky component of the perturbation and $\mathcal M_s$ is the modulation correction removing tangent Kerr modes.

The desired estimate is

$$
\mathcal E_s(t)+\int_0^t\mathcal D_s(\tau)\,d\tau
\le C_\chi\mathcal E_s(0)
  + C_\chi\int_0^t
       \|N[h,h](\tau)\|_{H^{s-1}_w}^2\,d\tau .
$$

If the gauge has a weak-null hierarchy, then

$$
\|N[h,h](t)\|_{H^{s-1}_w}
\le C_\chi \epsilon (1+t)^{-1-\eta}\mathcal E_s(t)^{1/2}.
$$

This gives the bootstrap closure

$$
\mathcal E_s(t)^{1/2}\le 2C_\chi\epsilon(1+t)^{-p_\chi}
\quad\Longrightarrow\quad
\mathcal E_s(t)^{1/2}\le C_\chi\epsilon(1+t)^{-p_\chi}
$$

for $\epsilon\le\epsilon_\chi$.

## 6. Conditional Theorem

**Theorem.**  
Fix $|a|<M$. If the Carter Calibration Lemma holds with constants compatible with the Chandrasekhar transformation and with a global weak-null Kerr modulation gauge, then smooth vacuum initial data sufficiently close to Kerr$(M,a)$ have a future exterior development with complete future null infinity and decay to a nearby Kerr$(M_f,a_f)$ spacetime.

**Proof sketch.**

1. The Carter Calibration Lemma supplies the missing trapped-superradiant local energy estimate.
2. Red-shift and exterior timelike currents supply the horizon and far-region estimates.
3. Chandrasekhar reconstruction transfers the estimate to gravitational perturbations.
4. The modulation gauge removes the four-dimensional Kerr tangent obstruction.
5. The weak-null hierarchy makes the quadratic source integrable.
6. A standard continuity argument gives global existence, decay, and convergence to Kerr.

## 7. The Exact Remaining Work

To turn this invented framework into a proof, one must actually prove:

- the Carter Calibration Lemma on the full trapped-superradiant set,
- the finite resonant-orbit certificate with rational or interval margins,
- the quantized commutator estimate without unabsorbable derivative loss,
- compatibility with the spin-$\pm2$ Teukolsky system,
- compatibility with nonlinear metric reconstruction and modulation.

This is the cleanest invented route I can formulate: the conjecture is reduced to a single new microlocal positivity theorem plus known-style nonlinear bootstrap machinery.
