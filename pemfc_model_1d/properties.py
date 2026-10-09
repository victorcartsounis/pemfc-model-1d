"""Constitutive relations, as SymPy expressions of the unknowns and parameters."""

import sympy as sp


def smooth_max0(z, width):
    """max{0, z}, rounded off over `width` around z = 0."""
    return (z + sp.sqrt(z**2 + width**2)) / 2


def smooth_step(z, width):
    """0 for z < 0 and 1 for z > 0, blended over `width` around z = 0."""
    return (1 + z / sp.sqrt(z**2 + width**2)) / 2


def arrhenius(p, activation_energy, T):
    return sp.exp(activation_energy / p.R * (1 / p.T_ref - 1 / T))


def saturation_pressure(T):
    """Water vapour saturation pressure in Pa (Antoine equation)."""
    return sp.exp(23.1963 - 3816.44 / (T - 46.13))


def water_viscosity(T):
    """Dynamic viscosity of liquid water in Pa s (Vogel equation)."""
    return 1e-3 * sp.exp(-3.63148 + 542.05 / (T - 144.15))


def capillary_pressure(s):
    """Capillary pressure in Pa as a function of the liquid saturation."""
    return (
        -0.00011 * sp.exp(-44.02 * (s - 0.496))
        + 278.3 * sp.exp(8.103 * (s - 0.496))
        - 191.8
    )


def reduced_saturation(p, s):
    """Share of the pore space above the immobile saturation, which is taken as
    s_C. Immobile water neither flows nor changes phase, so the reduced
    saturation is never negative."""
    return smooth_max0((s - p.s_C) / (1 - p.s_C), p.smoothing)


def hydraulic_permeability(p, s, kappa_abs):
    return (1e-6 + reduced_saturation(p, s) ** 3) * kappa_abs


def phase_change_rate(p, s, x_H2O, x_sat, T, C):
    """Condensation (positive) or evaporation (negative) in mol/(m^3 s)."""
    hertz_knudsen = sp.sqrt(p.R * T / (2 * sp.pi * p.M_w))
    s_red = reduced_saturation(p, s)
    gamma_e = p.k_e_factor * hertz_knudsen * p.a_lg * s_red
    gamma_c = p.k_c_factor * hertz_knudsen * p.a_lg * (1 - s_red)
    condensing = smooth_step(x_H2O / x_sat - 1, p.smoothing)
    return (gamma_e + (gamma_c - gamma_e) * condensing) * C * (x_H2O - x_sat)


def equilibrium_water_content(RH):
    """Ionomer water content in equilibrium with vapour at relative humidity RH."""
    return 0.043 + 17.81 * RH - 39.85 * RH**2 + 36.0 * RH**3


def water_volume_fraction(p, lam):
    return lam * p.V_w / (lam * p.V_w + p.V_m)


def proton_conductivity(p, lam, T, eps_i):
    f = water_volume_fraction(p, lam)
    percolation = smooth_max0(f - 0.06, 0.06 * p.smoothing) ** 1.5
    return eps_i**1.5 * p.sigma_p_ref * percolation * arrhenius(p, p.E_sigma_p, T)


def ionomer_water_diffusivity(p, lam, T, eps_i):
    """Diffusivity of dissolved water in the ionomer, in m^2/s."""
    fit = (3.842 * lam**3 - 32.03 * lam**2 + 67.74 * lam) / (
        lam**3 - 2.115 * lam**2 - 33.013 * lam + 103.37
    )
    return eps_i**1.5 * fit * 1e-10 * arrhenius(p, p.E_D_lam, T)


def drag_coefficient(lam):
    """Electro-osmotic drag coefficient (water molecules per proton)."""
    return 2.5 * lam / 22


def sorption_rate(p, lam, lam_eq, T):
    """Water sorption into the ionomer of a catalyst layer, in mol/(m^3 s):
    absorption below equilibrium, desorption above it."""
    desorbing = smooth_step(lam / lam_eq - 1, p.smoothing)
    prefactor = p.a_a + (p.a_d - p.a_a) * desorbing
    k = prefactor * water_volume_fraction(p, lam) * arrhenius(p, p.E_sorption, T)
    return k / (p.L_CL * p.V_m) * (lam_eq - lam)


def gas_diffusivity(p, D_ref, T, P, eps_p, tau, s):
    """Effective diffusivity in a partially flooded porous layer."""
    return eps_p / tau**2 * (1 - s) ** 3 * D_ref * (T / p.T_ref) ** 1.5 * p.P_ref / P


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
