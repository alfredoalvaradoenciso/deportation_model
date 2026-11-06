"""
Guatemala: Labor Market Absorption of Deportation Inflows  (JAX)
================================================================

Search-and-matching model of a dual labor market. DMP core with three departures:
  1. Frictionless informal sector with decreasing returns  -> residual absorber
  2. Endogenous participation margin                       -> remittance channel
  3. Exogenous population shock N (returnee inflow)        -> the experiment

Timing: SEQUENTIAL. All participants search; non-matchers work informally within
the same period. States: Formal (F), Informal (I), Out of labor force (O).
There is no unemployment stock, so searchers S = A - F = I.

Wages are FLEXIBLE. Formal absorption is capped by DECREASING RETURNS in the
formal sector (structurally low productivity / quasi-fixed capital).

Partial equilibrium, deterministic, quarterly. Steady-state comparative statics.

--------------------------------------------------------------------------
IMPLEMENTATION NOTES (JAX)
--------------------------------------------------------------------------
* Solved in TRANSFORMED coordinates so iterates are always interior:
      A = N*sigmoid(a),  F = A*sigmoid(f),  V = exp(v)
  This guarantees 0 < F < A <= N and V > 0 without clipping. Clipping inside a
  residual creates flat regions and silently corrupts the Jacobian.
* Job-finding rate uses the smooth urn-ball form p_F = 1 - exp(-M_raw/S) instead
  of min(M, S). jnp.minimum is a kink and Newton behaves badly across it.
* Newton via jax.jacfwd + jnp.linalg.solve inside lax.scan. Everything is jittable
  and differentiable, so we can later autodiff outcomes w.r.t. parameters.
* float64 enabled: the residuals span several orders of magnitude.

Requires: jax, jaxlib, numpy.
"""

from typing import NamedTuple, Dict
import functools

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# ============================================================================
# PARAMETERS
# ============================================================================

class ModelParams(NamedTuple):
    # --- Preferences and income ---
    sigma:   float = 1.5     # CRRA coefficient. Concavity drives the remittance channel.
    beta:    float = 0.96    # ANNUAL discount factor (~4% real rate)
    R:       float = 0.60    # Remittances per capita, ANNUAL flow
    mu_z:    float = 0.55    # Mean of exponential distribution of home production z

    # --- Formal sector ---
    A_F:     float = 0.55    # Formal productivity scale
    gamma_F: float = 0.30    # KEY. Decreasing returns: y_F(F) = A_F * F^(-gamma_F).
    c_F:     float = 0.55    # Vacancy posting cost per period
    delta_F: float = 0.22    # ANNUAL separation rate = 1-(1-0.06)^4
    A_M:     float = 0.45    # Matching efficiency
    alpha_M: float = 0.50    # Matching elasticity w.r.t. vacancies
    eps:     float = 2.00    # KEY. Firm-level labor supply elasticity (MONOPSONY).
                             # w_F = [eps/(1+eps)] * y_F, a markdown on marginal
                             # product. eps -> inf gives the competitive case
                             # w_F = y_F. Lower eps = more wage-setting power.
                             # NOTE: w_I does NOT enter w_F. Employers do not
                             # compete with the informal sector

    # --- Informal sector ---
    A_I:     float = 0.65    # Informal productivity scale
    eta_I:   float = 0.40    # KEY. Decreasing returns: y_I(I) = A_I * I^(1-eta_I),
                             # w_I = marginal product. This is the absorption margin.


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

def utility(c, p: ModelParams):
    """CRRA period utility. Concavity is what lets remittances move participation."""
    c = jnp.maximum(c, 1e-12)
    return jnp.where(
        jnp.abs(p.sigma - 1.0) < 1e-10,
        jnp.log(c),
        (jnp.power(c, 1.0 - p.sigma) - 1.0) / (1.0 - p.sigma),
    )


def inverse_utility(u_val, p: ModelParams):
    """Certainty-equivalent consumption delivering utility u_val."""
    safe = jnp.maximum(u_val * (1.0 - p.sigma) + 1.0, 1e-12)
    return jnp.where(
        jnp.abs(p.sigma - 1.0) < 1e-10,
        jnp.exp(u_val),
        jnp.power(safe, 1.0 / (1.0 - p.sigma)),
    )


def informal_wage(I, p: ModelParams):
    """w_I = d/dI [ A_I * I^(1-eta_I) ]. Falls as the informal sector absorbs workers."""
    return p.A_I * (1.0 - p.eta_I) * jnp.power(jnp.maximum(I, 1e-10), -p.eta_I)


def formal_productivity(F, p: ModelParams):
    """y_F = A_F * F^(-gamma_F). Marginal product falls as formal employment expands.
    This is what caps formal absorption."""
    return p.A_F * jnp.power(jnp.maximum(F, 1e-10), -p.gamma_F)


def participation_cutoff(w_I, w_F, p_F, p: ModelParams):
    """Cutoff z* on home-production value. Participate iff z <= z*.
        work: u(market income ),  market income = w_F w.p. p_F, else w_I
        home: u(z + R)
        z* solves u(z* + R) = E[u(market income )]
    """
    Eu = p_F * utility(w_F , p) + (1.0 - p_F) * utility(w_I , p)
    return jnp.maximum(inverse_utility(Eu, p) - p.R, 0.0)


# ============================================================================
# EQUILIBRIUM BLOCK
# ============================================================================
# Unknowns (transformed): x = [a, f, v]
#   A = N*sigmoid(a)        active labor force
#   F = A*sigmoid(f)        formal employment   (guarantees 0 < F < A)
#   V = exp(v)              vacancies
#   I = A - F = S           informal employment = searchers (sequential timing)

def unpack(x, p: ModelParams, N: float):
    a, f, v = x[0], x[1], x[2]
    A = N * jax.nn.sigmoid(a)
    F = A * jax.nn.sigmoid(f)
    V = jnp.exp(v)
    return A, F, V


def equilibrium_block(x, p: ModelParams, N: float) -> Dict[str, jnp.ndarray]:
    A, F, V = unpack(x, p, N)

    S = A - F                                     # searchers = informal employment
    I = S

    # --- Matching (smooth; no min() kink) ---
    m_raw = p.A_M * jnp.power(V, p.alpha_M) * jnp.power(S, 1.0 - p.alpha_M)
    p_F = -jnp.expm1(-m_raw / S)                  # = 1 - exp(-m_raw/S), in (0,1)
    M = p_F * S
    q_F = M / V

    # --- Prices ---
    w_I = informal_wage(I, p)
    y_F = formal_productivity(F, p)
    w_F = (p.eps / (1.0 + p.eps)) * y_F  # monopsony markdown on marginal product

    # --- Participation ---
    z_star = participation_cutoff(w_I, w_F, p_F, p)
    A_imp = N * (-jnp.expm1(-z_star / p.mu_z))    # exponential CDF at z_star

    # --- Value of a filled formal job ---
    J_F = (y_F / (1.0 + p.eps)) / (1.0 - p.beta * (1.0 - p.delta_F))

    return dict(A=A, F=F, V=V, I=I, S=S, O=N - A,
                p_F=p_F, q_F=q_F, M=M,
                w_I=w_I, w_F=w_F, y_F=y_F,
                z_star=z_star, A_imp=A_imp, J_F=J_F)


def residuals(x, p: ModelParams, N: float):
    b = equilibrium_block(x, p, N)
    return jnp.array([
        b["M"] - p.delta_F * b["F"],              # (1) hires = separations
        p.c_F - p.beta * b["q_F"] * b["J_F"],     # (2) free entry
        b["A_imp"] - b["A"],                      # (3) participation consistency
    ])

# ============================================================================
# NEWTON SOLVER
# ============================================================================

@functools.partial(jax.jit, static_argnums=(3,))
def _newton(x0, p: ModelParams, N: float, n_iter: int = 80):
    """Damped Newton with a trust radius. Returns final x and the residual-norm path."""
    def step(x, _):
        fx = residuals(x, p, N)
        Jm = jax.jacfwd(residuals)(x, p, N)
        dx = jnp.linalg.solve(Jm + 1e-12 * jnp.eye(3), fx)
        # Trust radius: cap the step length in transformed space.
        nrm = jnp.linalg.norm(dx)
        dx = jnp.where(nrm > 1.0, dx / nrm, dx)
        return x - dx, jnp.linalg.norm(fx)

    x, path = lax.scan(step, x0, None, length=n_iter)
    return x, path


def solve_steady_state(p: ModelParams, N: float = 1.0, x0=None, n_iter: int = 80):
    """Solve the steady state. Returns a dict of levels, rates and diagnostics."""
    if x0 is None:
        # Transformed-space start: A ~ 0.65N, F ~ 0.35A, V ~ 0.15
        x0 = jnp.array([jnp.log(0.65 / 0.35), jnp.log(0.35 / 0.65), jnp.log(0.15)])

    x, path = _newton(x0, p, N, n_iter)
    b = equilibrium_block(x, p, N)
    res = jnp.linalg.norm(residuals(x, p, N))

    out = {k: float(v) for k, v in b.items()}
    out.update(
        N=float(N),
        informality=float(b["I"] / (b["F"] + b["I"])),
        participation=float(b["A"] / N),
        residual=float(res),
        x=x,
        res_path=path,
    )
    return out


def is_valid(s: Dict, tol: float = 1e-9) -> bool:
    """A solution is usable only if it converged AND is economically sensible.
    The transforms guarantee interiority, so a bad solve shows up as a large
    residual rather than as a negative stock. """
    return bool(
        s["residual"] < tol
        and 0.0 < s["informality"] < 1.0
        and 0.0 < s["participation"] <= 1.0
        and 0.0 < s["p_F"] < 1.0
        and s["J_F"] > 0.0
        and s["w_F"] > s["w_I"]
    )


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiments(inflow: float = 0.05, remit_cut: float = 0.30):
    p0 = ModelParams()
    base = solve_steady_state(p0)

    print("=" * 78)
    print("GUATEMALA: LABOR MARKET ABSORPTION OF DEPORTATION INFLOWS")
    print("=" * 78)

    print("\n[0] SOLVER DIAGNOSTICS")
    print(f"  |residual|            : {base['residual']:.3e}")
    print(f"  stocks sum to N       : "
          f"{abs(base['F'] + base['I'] + base['O'] - base['N']) < 1e-9}")
    print(f"  interior job finding  : {0.0 < base['p_F'] < 1.0}")
    print(f"  economically valid    : {is_valid(base)}")
    if not is_valid(base):
        print("  STOP: baseline is not a valid equilibrium. Retune parameters.")
        print("  Everything below is meaningless until this line reads True.")

    print("\n[1] BASELINE STEADY STATE (population normalized to 1)")
    for k, lab in [("F", "Formal employment    F"), ("I", "Informal employment  I"),
                   ("O", "Out of labor force   O"), ("V", "Vacancies            V"),
                   ("informality", "Informality rate"),
                   ("participation", "Participation rate"),
                   ("w_I", "Informal wage    w_I"), ("w_F", "Formal wage      w_F"),
                   ("y_F", "Formal marg. prod y_F"),
                   ("p_F", "Job finding rate p_F")]:
        print(f"  {lab:<24}: {base[k]:.4f}")
    print("\n  TARGETS to hit before interpreting magnitudes:")
    print("    informality ~ 0.76 (ENEI),  participation ~ 0.60-0.65 (ENEI)")
    print("    formal-informal wage gap from ENEI,  F level from IGSS")

    # --- Scenarios ---
    p_rem = p0._replace(R=p0.R * (1.0 - remit_cut))
    dep  = solve_steady_state(p0,    N=1.0 + inflow, x0=base["x"])
    rem  = solve_steady_state(p_rem, N=1.0,          x0=base["x"])
    comb = solve_steady_state(p_rem, N=1.0 + inflow, x0=base["x"])

    print(f"\n[2] SCENARIO DECOMPOSITION "
          f"(inflow = {inflow:.0%} of population, remittances -{remit_cut:.0%})")
    hdr = f"  {'':<24}{'Base':>10}{'Deport':>10}{'Remit':>10}{'Both':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, lab in [("informality", "Informality rate"),
                   ("participation", "Participation rate"),
                   ("w_I", "Informal wage"), ("w_F", "Formal wage"),
                   ("F", "Formal employment"), ("I", "Informal employment"),
                   ("p_F", "Job finding rate")]:
        print(f"  {lab:<24}{base[k]:>10.4f}{dep[k]:>10.4f}"
              f"{rem[k]:>10.4f}{comb[k]:>10.4f}")
    print(f"  {'[valid]':<24}{str(is_valid(base)):>10}{str(is_valid(dep)):>10}"
          f"{str(is_valid(rem)):>10}{str(is_valid(comb)):>10}")

    if all(is_valid(s) for s in (base, dep, rem, comb)):
        d_dep = dep["informality"] - base["informality"]
        d_rem = rem["informality"] - base["informality"]
        d_tot = comb["informality"] - base["informality"]
        print(f"\n  Informality change, deportation channel : {d_dep:+.4f}")
        print(f"  Informality change, remittance channel  : {d_rem:+.4f}")
        print(f"  Informality change, combined            : {d_tot:+.4f}")
        print("\n  NOTE: if these are near zero or NEGATIVE, that is a real property")
        print("  of the model at these parameters, not a solver failure. Read the")
        print("  DIAGNOSIS section at the bottom of this file.")

    # --- Sensitivity over the two unidentified parameters ---
    print("\n[3] SENSITIVITY: change in informality under the deportation shock")
    print("    rows = eps (monopsony power), cols = gamma_F (formal returns)")
    epss = [1.00, 1.50, 2.00, 3.00, 5.00]
    gams = [0.10, 0.20, 0.30, 0.40, 0.50]
    print("         " + "".join(f"{g:>10.2f}" for g in gams))
    for e in epss:
        row = f"  {e:>5.2f}  "
        for g in gams:
            pp = p0._replace(eps=e, gamma_F=g)
            b = solve_steady_state(pp)
            if not is_valid(b):
                row += f"{'n/a':>10}"
                continue
            d = solve_steady_state(pp, N=1.0 + inflow, x0=b["x"])
            row += (f"{d['informality'] - b['informality']:>+10.4f}"
                    if is_valid(d) else f"{'n/a':>10}")
        print(row)
    print("  'n/a' = no valid equilibrium at those parameters, NOT a zero effect.")

    # --- Mechanism check ---
    print("\n[4] MECHANISM CHECK: shut down formal decreasing returns (gamma_F = 0)")
    p_flat = p0._replace(gamma_F=0.0)
    b0 = solve_steady_state(p_flat)
    if is_valid(b0):
        d0 = solve_steady_state(p_flat, N=1.0 + inflow, x0=b0["x"])
        if is_valid(d0):
            print(f"  gamma_F = {p0.gamma_F:.2f} : informality "
                  f"{base['informality']:.4f} -> {dep['informality']:.4f}  "
                  f"({dep['informality'] - base['informality']:+.4f})")
            print(f"  gamma_F = 0.00 : informality "
                  f"{b0['informality']:.4f} -> {d0['informality']:.4f}  "
                  f"({d0['informality'] - b0['informality']:+.4f})")
            print("  The gap between these two lines IS the contribution of formal")
            print("  decreasing returns. If it is small, gamma_F is not the binding")
            print("  constraint on absorption at this calibration.")
        else:
            print("  No valid shocked equilibrium at gamma_F = 0.")
    else:
        print("  No valid baseline at gamma_F = 0.")

    print("\n" + "=" * 78)
    print("PARAMETERS ARE ILLUSTRATIVE PLACEHOLDERS, NOT A CALIBRATION.")
    print("Calibrate A_I, A_F, c_F, A_M, mu_z, delta_F, R so that the BASELINE")
    print("reproduces ENEI informality and participation and the IGSS formal")
    print("employment level, before attaching meaning to any magnitude above.")
    print("=" * 78)


if __name__ == "__main__":
    run_experiments()
