# Renormalized Carter Transport Algebra
## A Breakthrough Candidate for Full-Subextremal Kerr Stability

## Status

This note proposes new mathematics. It is not a completed proof of nonlinear Kerr stability. The intended breakthrough is structural: reduce the trapped-superradiant sign problem to a finite algebraic positivity problem on a renormalized Carter transport algebra.

The central idea is to stop trying to find one global multiplier directly. Instead, build an algebra of transported observables whose averaged cone is positive on the trapped-superradiant set.

## 1. The New Algebra

Let

$$
\mathcal C=(E,L,Q)
$$

be the Kerr constants of motion on the characteristic set $p=0$. Let $K_{\mathrm{tsr}}$ denote the trapped-superradiant characteristic set.

The concrete radial input is the null Carter polynomial

$$
\Delta=r^2-2Mr+a^2,\qquad
P(r)=E(r^2+a^2)-aL,
$$

$$
\mathcal R(r;E,L,Q)=P(r)^2-\Delta\bigl(Q+(L-aE)^2\bigr).
$$

The trapped radial constraints are

$$
\rho_0=\mathcal R,\qquad \rho_1=\partial_r\mathcal R.
$$

On a chart with $E\ne0$ and $r\ne M$, write $\lambda=L/E$ and $\eta=Q/E^2$. The spherical-orbit equations $\mathcal R=\partial_r\mathcal R=0$ imply

$$
\lambda(r)=\frac{r^2+a^2}{a}-\frac{2r\Delta}{a(r-M)},
$$

$$
\eta(r)=\frac{4r^2\Delta}{(r-M)^2}-(\lambda(r)-a)^2.
$$

These formulas are the coordinate backbone of the certificate atlas. Each radial
chart must carry denominator certificates for the displayed divisions, and each
adjacent pair must carry a transition certificate proving equality of the shared
endpoint values of $\lambda$, $\eta$, the radial Hessian, and the horizon-frequency
data used in the superradiant factor. When two charts overlap on a positive-length
radial interval, the overlap certificate must recompute the Carter interval data
on that common interval and prove containment in both chart records. Polar,
equatorial, and $a=0$ degeneration charts must be handled separately by
equivalent variables. For the polar $L=0$ degeneration, the regular seed
constraint is $\lambda=0$. The spherical equations reduce, away from $r=M$, to
the cubic
$$
r^3-3Mr^2+a^2r+Ma^2=0,
$$
with
$$
\eta=\frac{2r(r^2+a^2)}{r-M}-a^2.
$$
For the spin-half atlas, the exterior polar root is isolated by the rational
interval $r/M\in[576643/200000,144161/50000]$, with certified positive $r-M$,
$\Delta$, $\eta$, and radial-Hessian interval. For the equatorial $Q=0$
degeneration, the regular seed constraint is $\eta=0$. It reduces to
$$
r^3-6Mr^2+9M^2r-4Ma^2=0.
$$
For the spin-half atlas, the prograde and retrograde exterior roots are isolated
by
$$
r/M\in[2347/1000,587/250],
\qquad
r/M\in[883/250,3533/1000],
$$
with certified positive $r-M$, $\Delta$, and radial-Hessian intervals. For
$a=0$, the regular seed variable is
$j=(L^2+Q)/E^2$, and the Schwarzschild photon-sphere chart is fixed by
$r=3M$, $j=27M^2$. The first transition blow-up chart uses
$$
\sigma=\frac{r-3M}{a},\qquad j=\lambda^2+\eta,
$$
and verifies that the pole in $j$ cancels exactly:
$$
j=
\frac{M^2a^2-6M^2r^2+2Ma^2r+a^2r^2+2r^4}{(r-M)^2}.
$$
Along $r=3M+a\sigma$, this gives
$$
\lambda\to-\frac92M\sigma,\qquad
\eta\to 27M^2-\frac{81}{4}M^2\sigma^2,\qquad
j\to27M^2,
$$
and the radial-Hessian scale tends to $54M^2$. An exact rational interval
certificate also verifies the local box $|a|/M\le 1/64$, $|\sigma|\le1$, where
$r/M\in[191/64,193/64]$, $r-M$ and $\Delta$ stay positive, and the intervals for
$j/M^2$ and the normalized radial Hessian contain $27$ and $54$. Thus the
zero-spin obstruction is no longer a formal division-by-$a$ singularity; it is
reduced to extending this local blow-up box to a complete small-spin cover and
proving overlap certificates with the nonzero-spin atlas.

Define the **renormalized Carter variables**

$$
\mathsf e = E,\qquad
\mathsf l = E-\Omega_H L,\qquad
\mathsf q = \frac{Q+\alpha_\chi L^2+\beta_\chi E^2}{\Delta_{\mathrm{trap}}(\chi)},
$$

where $\Delta_{\mathrm{trap}}(\chi)$ is the degeneracy scale of the trapped radial Hessian and $\alpha_\chi,\beta_\chi$ are chosen so that $\mathsf q$ remains bounded and nonnegative on trapped null geodesics as $\chi\to1$ for fixed subextremal $\chi<1$.

For certificate purposes one may take

$$
\Delta_{\mathrm{trap}}(\chi,\rho)
=1+\left|\partial_r^2\mathcal R(r_{\mathrm{trap}};E,L,Q)\right|
$$

or any rational interval enclosure bounded above and below on the chart.

The **renormalized Carter transport algebra** $\mathfrak A_\chi$ is the smallest real filtered algebra of symbols on a conic neighborhood of $K_{\mathrm{tsr}}$ containing

$$
1,\quad \mathsf e,\quad \mathsf l,\quad \mathsf q,\quad r-r_{\mathrm{trap}}(\mathcal C),\quad H_p(r-r_{\mathrm{trap}}(\mathcal C)),
$$

and closed under:

- multiplication,
- Poisson bracket with $p$,
- torus averaging along the Kerr trapped flow,
- division by nonzero averaged elliptic factors with interval-certified lower bounds.

The filtration is by total Carter degree plus one radial-transport degree for each application of $H_p$.

## 2. Positive Cone

Define the **Carter transport cone** $\mathfrak A_\chi^+$ to consist of all symbols admitting a representation

$$
a
= \sum_j s_j^2
  + \sum_k \lambda_k \mathsf l^2 t_k^2
  + H_p g
  + \sum_m \mu_m \rho_m^2
  - R,
$$

where:

- $s_j,t_k,g,\rho_m\in\mathfrak A_\chi$,
- $\lambda_k,\mu_m\ge0$,
- $\rho_m$ define the trapped constraints,
- $R$ is a lower-order remainder absorbable by red-shift, exterior ellipticity, or nonsuperradiant Morawetz estimates.

The term $\mathsf l^2=(E-\Omega_H L)^2$ is the superradiant compensator. It is small on the horizon generator but positive away from the superradiant threshold. The $H_p g$ term is the exact transport correction that turns orbit-average positivity into pointwise commutator positivity.

## 3. Breakthrough Lemma

**Renormalized Carter Cone Lemma.**  
For each fixed $|a|<M$, the spin-$2$ gravitational energy density $q_{\mathrm{grav}}$ restricted to a conic neighborhood of $K_{\mathrm{tsr}}$ belongs to the interior of the Carter transport cone:

$$
q_{\mathrm{grav}}\in \operatorname{int}\mathfrak A_\chi^+.
$$

Equivalently, there exist $\delta_\chi>0$ and $b,g,R\in\mathfrak A_\chi$ such that

$$
H_p b
+ \mathsf l^2
+ \operatorname{dist}(\rho,K_{\mathrm{tsr}})^2
- R
\ge \delta_\chi q_{\mathrm{grav}}.
$$

This lemma is stronger than the Carter Calibration Lemma because it gives an algebraic certificate for the commutator symbol rather than merely asserting existence of a symbol.

## 4. Why This Is a Plausible Breakthrough

Kerr trapping is integrable. Every trapped orbit lies on an invariant torus, and the hard obstruction is the sign of averaged densities on these tori. The algebra $\mathfrak A_\chi$ is designed so that:

1. nonresonant torus oscillations are exact $H_p$-coboundaries,
2. resonant tori reduce to finitely many Fourier modes,
3. superradiant negativity is measured by the single square $\mathsf l^2$,
4. near-extremal degeneracy is normalized into $\mathsf q$,
5. the final positivity claim becomes an SOS-plus-transport certificate.

The old obstruction was analytic and global. The new obstruction is algebraic and local on the integrable torus fibration.

## 5. Certificate Form

A proof of the Renormalized Carter Cone Lemma should produce:

$$
q_{\mathrm{grav}}-\delta_\chi^{-1}
\left(
H_p b+\mathsf l^2+\operatorname{dist}(\rho,K_{\mathrm{tsr}})^2
\right)
=
\sum_j s_j^2
+\sum_k \lambda_k\mathsf l^2 t_k^2
+\sum_m \mu_m\rho_m^2
-R.
$$

To be machine-checkable, every coefficient is represented by rational intervals in variables

$$
(\mathsf e,\mathsf l,\mathsf q,\vartheta_1,\vartheta_2)
$$

on a finite torus atlas. Resonant strata are handled by substituting rational winding ratios and checking the remaining polynomial inequalities.
The current radial scaffold verifies exact Carter identities, denominator
nonvanishing, contiguous radial coverage, adjacent endpoint compatibility, and
positive-length overlap compatibility for toy patch pairs. It now verifies the
exact spin-half polar $L=0$ seed chart, the Schwarzschild $a=0$ photon-sphere
seed chart, the spin-half equatorial $Q=0$ prograde and retrograde seed charts,
a blow-up transition seed using $\sigma=(r-3M)/a$ and $j=\lambda^2+\eta$, and a
first rational interval box around it. A degenerate-seed bridge certificate now
anchors the polar seed and the two equatorial seeds inside verified spin-half
generic radial charts, recomputes the nonzero-spin Carter interval data on those
seed intervals, and proves that the regular seed data are contained in the
generic interval data. A local angular-corridor certificate then regularizes the
degenerate angular potential: near the polar family it uses
$\lambda^2=\sin^2\theta\,\mu^2$, and near the equatorial family it uses
$\cos^2\theta=\eta\zeta$. Exact interval arithmetic proves positivity of the
regularized angular potential on two polar one-sided corridors and on the
prograde/retrograde equatorial eta-corridors. A further angular-overlap strip
certificate proves that the corridor variables glue to the ordinary angular
Carter chart on strips where the ordinary denominators are bounded away from
zero: $\sin^2\theta\in[1/64,1/16]$ on the polar side and
$\eta\in[1/8192,1/4096]$ on the equatorial side. An angular patch-coverage
certificate then proves the polar seed-tube cover
$[0,1/16]\cup[1/64,1]=[0,1]$ in $v=\sin^2\theta$, with positive ordinary
angular potential on the annulus, and extends the equatorial blow-up coordinate
to $\zeta\in[0,1/128]$ with positive bracket and ordinary-overlap margins on
both branches. A seed-tube thickening certificate then follows the chain back
to the radial bridge: it proves a two-sided polar tube
$|\lambda|\le 1/6000$ with $\mu^2\le1/2250000$, ordinary-annulus positivity for
the full bridge lambda interval, and equatorial tubes
$\eta\le1/1024$, $\zeta\le1/48$ with positive blow-up and ordinary-overlap
margins on both branches. A local generic-overlap certificate now adds the
inverse-coordinate doorways from these seed tubes back to ordinary Carter
angular variables: on the polar side,
$|\lambda|\le1/12000$ and $\sin^2\theta\in[1/64,1/16]$ give
$\mu^2=\lambda^2/\sin^2\theta\le1/2250000$ with positive denominator and
positive ordinary/regularized angular potentials; on the equatorial side,
$\eta\in[1/8192,1/1024]$ and
$\cos^2\theta\in[0,1/393216]$ give
$\zeta=\cos^2\theta/\eta\le1/48$ with positive transition Jacobian and
positive ordinary/regularized angular potentials on both branches. The
remaining polar and equatorial work is to enlarge these local seed doorways
into rational angular-torus neighborhoods and verify complete positive-length
angular overlaps with generic charts. The remaining zero-spin work is to extend that local box to a complete small-spin trapped-set cover and verify full overlaps
between the small-spin blow-up charts and the nonzero-spin Carter atlas. The
scaffold also includes an exact rational Gram checker for PSD witnesses, but it
does not yet supply a Kerr-derived SOS/Gram cone membership certificate. The first
constructive companion is a sparse rational square-extraction solver: it can
produce small Gram witnesses for protocol targets, while a full proof still
requires a real Kerr-derived semidefinite/SOS solve on every chart.

## 6. Quantization Principle

The algebraic cone should quantize through a **Carter-Wick quantization**:

$$
\operatorname{Op}_{CW}:\mathfrak A_\chi\to\Psi^0_{\mathrm{Kerr}},
$$

with three properties:

1. squares quantize to positive operators up to lower-order remainders,
2. $H_p b$ quantizes to $i[P,\operatorname{Op}_{CW}(b)]$ up to controllable subprincipal terms,
3. torus-averaged resonant certificates remain positive under sharp Gårding with constants uniform on each fixed subextremal compact spin interval.

Then the cone certificate implies

$$
\operatorname{Re}\langle i[P,B]u,u\rangle
+\mathcal F_{\mathcal H^+}[u]
+\mathcal F_{\mathcal I^+}[u]
\ge
c_\chi\|u\|_{LE^1_{\mathrm{tsr}}}^2
-C_\chi\|u\|_{LE^0_{\mathrm{lower}}}^2.
$$

This is exactly the trapped-superradiant estimate needed by the main manuscript.

## 7. Nonlinear Consequence

If the Renormalized Carter Cone Lemma and Carter-Wick quantization principle are proved, then:

1. the Carter Calibration Lemma follows,
2. the trapped-superradiant branch of the Morawetz estimate closes,
3. the conditional nonlinear bootstrap in the manuscript applies,
4. small exterior perturbations of fixed subextremal Kerr decay to a nearby Kerr exterior, provided the global weak-null modulation gauge is available.

## 8. The Actual Breakthrough Target

The single finite target is:

> Produce an interval-certified SOS-plus-transport representation of $q_{\mathrm{grav}}$ in $\mathfrak A_\chi^+$ on a finite atlas of the trapped-superradiant torus fibration.

That is the strongest form of the new idea. It changes the problem from "find a miraculous global energy" to "check membership in a finitely generated positive cone adapted to the Carter flow."
