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
}


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str  # "GDL", "CL" or "PEM"
    side: str  # "A" (anode), "C" (cathode) or "" for the membrane
    variables: tuple[str, ...]


LAYERS = (
    Layer("AGDL", "GDL", "A", ("phi_e",)),
    Layer("ACL", "CL", "A", ("phi_e", "phi_p")),
    Layer("PEM", "PEM", "", ("phi_p",)),
    Layer("CCL", "CL", "C", ("phi_e", "phi_p")),
    Layer("CGDL", "GDL", "C", ("phi_e",)),
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
    """Solve the flux law `flux = law(gradient)` for the gradient."""
    return sp.solve(sp.Eq(flux, law(_GRADIENT)), _GRADIENT)[0]


def _reaction_rate(layer, p, u):
    """Volumetric current density of the half-reaction in a catalyst layer."""
    T = u["T"]
    P = gas_pressure(p, layer.side)
    reactant = u["x_H2"] if layer.side == "A" else u["x_O2"]
    dphi = u["phi_e"] - u["phi_p"]
    dphi0 = pr.reversible_potential(p, layer.side, T, reactant * P)
    eta = dphi - dphi0 if layer.side == "A" else dphi0 - dphi
    i0a = pr.volumetric_exchange_current(p, layer.side, T, reactant * P)
    return pr.butler_volmer(p, i0a, eta, T)


def equations(layer, p, u, j):
    """du/dx and dj/dx of every unknown of `layer`.

    `u` maps every unknown to its value in this layer (a symbol where it is
    solved for, a fixed expression where it is not); `j` maps the solved
    unknowns to their fluxes.
    """
    T = u["T"]
    gradient = {}
    source = dict.fromkeys(layer.variables, 0)

    if "phi_e" in layer.variables:
        sigma_e = getattr(p, f"sigma_e_{layer.kind}")
        gradient["phi_e"] = _gradient(j["phi_e"], lambda g: -sigma_e * g)
    if "phi_p" in layer.variables:
        eps_i = getattr(p, f"eps_i_{layer.kind}")
        sigma_p = pr.proton_conductivity(p, u["lam"], T, eps_i)
        gradient["phi_p"] = _gradient(j["phi_p"], lambda g: -sigma_p * g)

    if layer.kind == "CL":
        i = _reaction_rate(layer, p, u)
        sign = 1 if layer.side == "C" else -1
        source["phi_e"] = sign * i
        source["phi_p"] = -sign * i

    return {v: (gradient[v], source[v]) for v in layer.variables}
