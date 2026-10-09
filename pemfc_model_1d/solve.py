"""Polarization curves: solve the model over a descending sweep of cell voltages."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from pemfc_model_1d import layers as ly
from pemfc_model_1d.assembly import build
from pemfc_model_1d.parameters import Parameters


@dataclass(frozen=True)
class LayerProfile:
    x: np.ndarray  # m, measured from the anode GDL / gas channel face
    values: dict[str, np.ndarray]  # every unknown and flux of the layer, SI


@dataclass(frozen=True)
class OperatingPoint:
    voltage: float  # V
    current_density: float  # A/m^2
    layers: dict[str, LayerProfile]
    nodes: int


@dataclass(frozen=True)
class Sweep:
    points: tuple[OperatingPoint, ...]

    @property
    def voltages(self):
        return np.array([pt.voltage for pt in self.points])

    @property
    def current_densities(self):
        return np.array([pt.current_density for pt in self.points])


def solve(
    voltages,
    params=None,
    *,
    tol=1e-3,
    max_nodes=5000,
    initial_nodes=20,
    max_step=0.1,
    min_step=1e-3,
    step_growth=1.5,
):
    """Solve at each of `voltages` (V, strictly descending).

    Between requested voltages, intermediate voltages are solved as needed by
    continuation: the step shrinks on failure and grows on success, and each
    initial guess is extrapolated from the last two solutions. If the step
    falls below `min_step`, the sweep stops and returns the points reached.
    """
    voltages = [float(v) for v in voltages]
    if any(b >= a for a, b in zip(voltages, voltages[1:], strict=False)):
        raise ValueError("voltages must be strictly descending")

    params = params or Parameters()
    model = build()
    values = params.values()

    def attempt(U, x, y):
        args = (U, *values)
        return solve_bvp(
            lambda _, y: model.rhs(y, args),
            lambda ya, yb: model.bc(ya, yb, args),
            x,
            y,
            fun_jac=lambda _, y: model.rhs_jacobian(y, args),
            bc_jac=lambda ya, yb: model.bc_jacobian(ya, yb, args),
            tol=tol,
            max_nodes=max_nodes,
        )

    history = []  # the last two converged (voltage, result) pairs
    points = []
    step = max_step
    for target in voltages:
        while not history or history[-1][0] > target:
            U = target if not history else max(target, history[-1][0] - step)
            result = attempt(U, *_predict(model, history, U, values, initial_nodes))
            if result.success:
                history = [*history[-1:], (U, result)]
                step = min(step * step_growth, max_step)
            elif not history:
                raise RuntimeError(f"no solution at the first voltage {U} V")
            else:
                step /= 2
                if step < min_step:
                    warnings.warn(
                        f"continuation stalled below {history[-1][0]:.4f} V: "
                        f"{result.message}",
                        stacklevel=2,
                    )
                    return Sweep(tuple(points))
        points.append(_operating_point(model, params, target, history[-1][1]))
    return Sweep(tuple(points))


def _predict(model, history, U, values, initial_nodes):
    """Initial mesh and guess at voltage U."""
    if not history:
        x = np.linspace(0, 1, initial_nodes)
        y = np.repeat(model.initial_guess((U, *values))[:, None], x.size, axis=1)
        return x, y
    # solve_bvp only ever adds nodes, so features that have moved on would keep
    # their refinement; every other node is dropped and refined again if needed.
    U1, r1 = history[-1]
    x = np.append(r1.x[:-1:2], 1.0)
    if len(history) == 1:
        return x, r1.sol(x)
    U0, r0 = history[0]
    return x, r1.sol(x) + (U - U1) / (U1 - U0) * (r1.sol(x) - r0.sol(x))


def _operating_point(model, params, U, result):
    y = result.y * model.scale[:, None]
    layers, x0 = {}, 0.0
    for block in model.blocks:
        L = ly.thickness(block.layer, params)
        profile = {}
        for v in block.layer.variables:
            profile[v] = y[block.value(v)]
            profile[ly.VARIABLES[v][0]] = y[block.flux(v)]
        layers[block.layer.name] = LayerProfile(x0 + L * result.x, profile)
        x0 += L
    cathode = model.blocks[-1]
    return OperatingPoint(
        voltage=U,
        current_density=float(y[cathode.flux("phi_e"), -1]),
        layers=layers,
        nodes=result.x.size,
    )
