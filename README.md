# pemfc-model-1d

A one-dimensional, steady-state, non-isothermal, two-phase model of the
membrane electrode assembly of a low-temperature PEM fuel cell. Its base
formulation follows

> R. Vetter, J. O. Schumacher, *Free open reference implementation of a
> two-phase PEM fuel cell model*, Computer Physics Communications **234**
> (2019) 223–234. <https://doi.org/10.1016/j.cpc.2018.07.023>

It resolves the through-plane transport of electrons, protons, heat, dissolved
water, water vapour, hydrogen, oxygen and liquid water across five layers:
anode GDL, anode catalyst layer, membrane, cathode catalyst layer and cathode
GDL.

## How it is solved

The equations are written in SymPy as flux laws and conservation laws, the
way the paper states them. SymPy solves each flux law for the gradient, scales
every unknown to order one and derives the exact Jacobians, which are
lambdified once per process and handed to `scipy.integrate.solve_bvp`. Each
layer is mapped onto its own coordinate in [0, 1] and carries only the
unknowns that exist in it; continuity at the interfaces becomes boundary
conditions.

The switches in the constitutive relations (absorption/desorption,
evaporation/condensation, the percolation threshold of the proton
conductivity, the immobile liquid saturation, and the dead zone of a starved
catalyst layer) are smoothed over a width set by `Parameters.smoothing`.

A polarization curve is computed by continuation from high to low cell voltage,
with an adaptive step and a secant predictor between the requested voltages.

## Installation

```bash
poetry install
```

## Usage

```bash
poetry run python scripts/polarization_curve.py
```

sweeps the base case from 1.20 V to 0.05 V, prints the setup and sweep times
and the polarization curve, and writes figures to `results/`.

From Python:

```python
from pemfc_model_1d.parameters import Parameters
from pemfc_model_1d.solve import solve

sweep = solve([1.0, 0.8, 0.6], Parameters(RH_C=0.5))
sweep.voltages              # V
sweep.current_densities     # A/m^2
sweep.points[-1].layers["PEM"].values["lam"]   # water content across the membrane
```

All quantities are in SI units. Positions are measured from the anode GDL /
gas channel face.
