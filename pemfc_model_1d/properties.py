"""Constitutive relations, as SymPy expressions of the unknowns and parameters."""

import sympy as sp


def smooth_max0(z, width):
    """max{0, z}, rounded off over `width` around z = 0."""
    return (z + sp.sqrt(z**2 + width**2)) / 2


def arrhenius(p, activation_energy, T):
    return sp.exp(activation_energy / p.R * (1 / p.T_ref - 1 / T))


def saturation_pressure(T):
    """Water vapour saturation pressure in Pa (Antoine equation)."""
    return sp.exp(23.1963 - 3816.44 / (T - 46.13))


def equilibrium_water_content(RH):
    """Ionomer water content in equilibrium with vapour at relative humidity RH."""
    return 0.043 + 17.81 * RH - 39.85 * RH**2 + 36.0 * RH**3


def water_volume_fraction(p, lam):
    return lam * p.V_w / (lam * p.V_w + p.V_m)


def proton_conductivity(p, lam, T, eps_i):
    f = water_volume_fraction(p, lam)
    percolation = smooth_max0(f - 0.06, 0.06 * p.smoothing) ** 1.5
    return eps_i**1.5 * p.sigma_p_ref * percolation * arrhenius(p, p.E_sigma_p, T)


def reversible_potential(p, side, T, partial_pressure):
    """Share of the reversible cell potential carried by one catalyst layer,
    given the partial pressure of its reactant (H2 or O2)."""
    if side == "A":
        return -T * p.dS_HOR / (2 * p.F) - p.R * T / (2 * p.F) * sp.log(
            partial_pressure / p.P_ref
        )
    return -(p.dH - T * p.dS_ORR) / (2 * p.F) + p.R * T / (4 * p.F) * sp.log(
        partial_pressure / p.P_ref
    )


def volumetric_exchange_current(p, side, T, partial_pressure):
    """Exchange current density times platinum surface density, in A/m^3."""
    if side == "A":
        return p.a_ACL * p.i0_HOR * arrhenius(p, p.E_HOR, T)
    return (
        p.a_CCL
        * p.i0_ORR
        * (partial_pressure / p.P_ref) ** p.order_ORR
        * arrhenius(p, p.E_ORR, T)
    )


def butler_volmer(p, i0a, eta, T):
    """Volumetric reaction rate in A/m^3 for the activation overpotential eta."""
    f = 2 * p.F / (p.R * T)
    return i0a * (sp.exp(p.beta * f * eta) - sp.exp(-(1 - p.beta) * f * eta))
