"""Model parameters: physical constants, material properties and operating
conditions of the base case, all in SI units."""

from dataclasses import astuple, dataclass, fields

import sympy as sp


@dataclass(frozen=True)
class Parameters:
    # Physical constants
    F: float = 96485.33212  # C/mol
    R: float = 8.314462618  # J/(mol K)
    P_ref: float = 101325.0  # Pa
    T_ref: float = 353.15  # K, reference temperature of all Arrhenius laws
    M_w: float = 18e-3  # kg/mol, molar mass of water

    # Reaction thermodynamics and kinetics
    dH: float = -285.83e3  # J/mol, enthalpy of formation of liquid water
    dS_HOR: float = 0.104  # J/(mol K)
    dS_ORR: float = -163.3  # J/(mol K)
    i0_HOR: float = 0.27e4  # A/m^2 Pt
    E_HOR: float = 16e3  # J/mol
    i0_ORR: float = 2.45e-4  # A/m^2 Pt
    E_ORR: float = 67e3  # J/mol
    order_ORR: float = 0.54  # reaction order in the oxygen partial pressure
    a_ACL: float = 1e7  # m^2 Pt/m^3
    a_CCL: float = 3e7  # m^2 Pt/m^3
    beta: float = 0.5  # symmetry factor

    # Ionomer
    V_m: float = 1020 / 1.97 * 1e-6  # m^3/mol, equivalent volume of dry Nafion
    V_w: float = 18 / 0.978 * 1e-6  # m^3/mol, molar volume of liquid water
    sigma_p_ref: float = 116.0  # S/m
    E_sigma_p: float = 15e3  # J/mol

    # Layer thicknesses
    L_GDL: float = 160e-6  # m
    L_CL: float = 10e-6  # m
    L_PEM: float = 25e-6  # m

    # Material properties
    eps_i_CL: float = 0.3  # ionomer volume fraction
    eps_i_PEM: float = 1.0
    sigma_e_GDL: float = 1250.0  # S/m
    sigma_e_CL: float = 350.0  # S/m

    # Operating conditions
    P_A: float = 1.5e5  # Pa
    P_C: float = 1.5e5  # Pa
    RH_A: float = 0.9
    RH_C: float = 0.9
    s_C: float = 0.12  # liquid saturation at the cathode GDL / gas channel face
    T_A: float = 343.15  # K
    T_C: float = 343.15  # K
    alpha_H2: float = 1.0  # hydrogen mole fraction in the dry fuel
    alpha_O2: float = 0.21  # oxygen mole fraction in the dry oxidant

    # Width of the smooth replacement of every switch and max{0, .} in the
    # constitutive relations, relative to the scale of the switching quantity.
    smoothing: float = 1e-3

    @classmethod
    def symbols(cls) -> "Parameters":
        """The same parameter set with a SymPy symbol in place of every value."""
        return cls(**{f.name: sp.Symbol(f.name) for f in fields(cls)})

    def values(self) -> tuple:
        return astuple(self)
