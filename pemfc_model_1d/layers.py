"""The five MEA layers and the transport equations that hold in each.

Every unknown u obeys a flux law j = j(u, du/dx) and a conservation law
dj/dx = S. Both are written here as in the paper; the flux law is solved for
du/dx symbolically, which turns each pair into two first-order ODEs.
"""

from dataclasses import dataclass

import sympy as sp

from pemfc_model_1d import properties as pr

# Unknown -> (name of its flux, scale of the unknown, scale of the flux), in SI.
VARIABLES = {
    "phi_e": ("j_e", 1.0, 1e4),
    "phi_p": ("j_p", 1.0, 1e4),
    "T": ("j_T", 100.0, 1e4),
    "lam": ("j_lam", 10.0, 0.1),
    "x_H2O": ("j_H2O", 1.0, 0.1),
    "x_H2": ("j_H2", 1.0, 0.1),
    "x_O2": ("j_O2", 1.0, 0.1),
    "s": ("j_s", 0.1, 0.1),
}


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str  # "GDL", "CL" or "PEM"
    side: str  # "A" (anode), "C" (cathode) or "" for the membrane
    variables: tuple[str, ...]


LAYERS = (
    Layer("AGDL", "GDL", "A", ("phi_e", "T", "x_H2O", "x_H2")),
    Layer("ACL", "CL", "A", ("phi_e", "phi_p", "T", "lam", "x_H2O", "x_H2")),
    Layer("PEM", "PEM", "", ("phi_p", "T", "lam")),
    Layer("CCL", "CL", "C", ("phi_e", "phi_p", "T", "lam", "x_H2O", "x_O2", "s")),
    Layer("CGDL", "GDL", "C", ("phi_e", "T", "x_H2O", "x_O2", "s")),
)


def thickness(layer, p):
    return getattr(p, f"L_{layer.kind}")


def gas_pressure(p, side):
    return p.P_A if side == "A" else p.P_C


def channel_state(p, side, U):
    """Values imposed at the gas channel and plate on `side`."""
    T = p.T_A if side == "A" else p.T_C
    RH = p.RH_A if side == "A" else p.RH_C
    x_H2O = RH * pr.saturation_pressure(T) / gas_pressure(p, side)
    if side == "A":
        return {"phi_e": 0, "T": T, "x_H2O": x_H2O, "x_H2": p.alpha_H2 * (1 - x_H2O)}
    return {
        "phi_e": U,
        "T": T,
        "x_H2O": x_H2O,
        "x_O2": p.alpha_O2 * (1 - x_H2O),
        "s": p.s_C,
    }


def uniform_state(p, side, U):
    """Flat profile of every unknown on `side`, used as the initial guess."""
    state = channel_state(p, side, U) if side else {}
    state |= {
        "phi_p": 0,
        "T": (p.T_A + p.T_C) / 2,
        "lam": pr.equilibrium_water_content(1),
    }
    return state


_GRADIENT = sp.Symbol("gradient")


def _gradient(flux, law):
    """Solve the flux law `flux = law(gradient)`, which is linear in the
    gradient, for the gradient."""
    expr = law(_GRADIENT)
    slope = sp.diff(expr, _GRADIENT)
    assert not slope.has(_GRADIENT), "flux law is not linear in the gradient"
    return (flux - expr.subs(_GRADIENT, 0)) / slope


def _reaction_rate(layer, p, u):
    """Volumetric current density of the half-reaction in a catalyst layer,
    and its activation overpotential."""
    T = u["T"]
    P = gas_pressure(p, layer.side)
    if layer.side == "A":
        x, x_dry = u["x_H2"], p.alpha_H2
    else:
        x, x_dry = u["x_O2"], p.alpha_O2
    # The rate grows like x**0.79, so a starved catalyst layer has a dead zone
    # where x reaches zero and the rate is not differentiable. Evaluating it at
    # x_hat >= width and scaling by x / x_hat makes it linear in x near zero.
    x_hat = sp.sqrt(x**2 + (p.smoothing * x_dry) ** 2)
    dphi = u["phi_e"] - u["phi_p"]
    dphi0 = pr.reversible_potential(p, layer.side, T, x_hat * P)
    eta = dphi - dphi0 if layer.side == "A" else dphi0 - dphi
    i0a = pr.volumetric_exchange_current(p, layer.side, T, x_hat * P)
    return x / x_hat * pr.butler_volmer(p, i0a, eta, T), eta


def equations(layer, p, u, j):
    """du/dx and dj/dx of every unknown of `layer`.

    `u` maps every unknown to its value in this layer (a symbol where it is
    solved for, a fixed expression where it is not); `j` maps the solved
    unknowns to their fluxes.
    """
    T = u["T"]
    gradient = {}
    source = dict.fromkeys(layer.variables, 0)

    # Flux laws
    k = getattr(p, f"k_{layer.kind}")
    gradient["T"] = _gradient(j["T"], lambda g: -k * g)

    if layer.kind != "PEM":  # electron conducting
        sigma_e = getattr(p, f"sigma_e_{layer.kind}")
        gradient["phi_e"] = _gradient(j["phi_e"], lambda g: -sigma_e * g)

    if layer.kind != "GDL":  # ionomer
        lam = u["lam"]
        eps_i = getattr(p, f"eps_i_{layer.kind}")
        sigma_p = pr.proton_conductivity(p, lam, T, eps_i)
        gradient["phi_p"] = _gradient(j["phi_p"], lambda g: -sigma_p * g)
        D_lam = pr.ionomer_water_diffusivity(p, lam, T, eps_i)
        drag = pr.drag_coefficient(lam) / p.F * j["phi_p"]
        gradient["lam"] = _gradient(j["lam"], lambda g: -D_lam / p.V_m * g + drag)

    if layer.side:  # gas filled pores
        P = gas_pressure(p, layer.side)
        C = P / (p.R * T)
        eps_p = getattr(p, f"eps_p_{layer.kind}")
        tau = getattr(p, f"tau_{layer.kind}")
        s = u.get("s", 0)  # liquid water is only modelled on the cathode side
        D_ref = {
            "x_H2O": p.D_H2O_A if layer.side == "A" else p.D_H2O_C,
            "x_H2": p.D_H2,
            "x_O2": p.D_O2,
        }
        for x in ("x_H2O", "x_H2", "x_O2"):
            if x in layer.variables:
                D = pr.gas_diffusivity(p, D_ref[x], T, P, eps_p, tau, s)
                gradient[x] = _gradient(j[x], lambda g, D=D: -C * D * g)

    if "s" in layer.variables:  # liquid water in the cathode pores
        kappa = pr.hydraulic_permeability(p, s, getattr(p, f"kappa_{layer.kind}"))
        w = sp.Dummy()
        dpc_ds = sp.diff(pr.capillary_pressure(w), w).subs(w, s)
        mobility = kappa / (pr.water_viscosity(T) * p.V_w) * dpc_ds
        gradient["s"] = _gradient(j["s"], lambda g: -mobility * g)

    # Sources
    for phi in ("phi_e", "phi_p"):  # Joule heating
        if phi in layer.variables:
            source["T"] -= j[phi] * gradient[phi]

    if layer.kind == "CL":
        i, eta = _reaction_rate(layer, p, u)
        S_F = i / (2 * p.F)
        sign = 1 if layer.side == "C" else -1
        source["phi_e"] = sign * i
        source["phi_p"] = -sign * i
        dS = p.dS_HOR if layer.side == "A" else p.dS_ORR
        source["T"] += i * eta - S_F * T * dS
        if layer.side == "A":
            source["x_H2"] = -S_F
        else:
            source["x_O2"] = -S_F / 2
            source["lam"] = S_F  # product water appears dissolved in the ionomer

        x_sat = pr.saturation_pressure(T) / P
        lam_eq = pr.equilibrium_water_content(u["x_H2O"] / x_sat)
        S_ad = pr.sorption_rate(p, lam, lam_eq, T)
        source["lam"] += S_ad
        source["x_H2O"] -= S_ad
        source["T"] += p.H_ad * S_ad

    if "s" in layer.variables:
        x_sat = pr.saturation_pressure(T) / P
        S_ec = pr.phase_change_rate(p, s, u["x_H2O"], x_sat, T, C)
        source["s"] += S_ec
        source["x_H2O"] -= S_ec
        source["T"] += p.H_ec * S_ec

    return {v: (gradient[v], source[v]) for v in layer.variables}
